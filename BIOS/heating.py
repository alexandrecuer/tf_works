import numpy as np
import random
import math
#import tensorflow as tf
from datetime import datetime
import struct
import matplotlib.pylab as plt
import time

winterStart=1539950400

# convert a unix time stamp expressed in seconds to something human readable
def humanDate(uts):
    human=datetime.utcfromtimestamp(uts).strftime('%Y-%m-%d %H:%M:%S')
    return human

# is this really needed or would we go for struct.unpack all the time ?
def readBytes(nb,file):
    bytes = file.read(4)
    value=int.from_bytes(bytes, byteorder='little', signed=False)
    return value

# for debugging in case
def checkPoints(nb,position,interval):
    with open("phpfina/"+str(nb)+".dat", "rb") as ts:
        for i in range(300):
            nbPtInHour=60*60/interval
            offset=int((position+i)*4)
            print(offset)
            ts.seek(offset, 0)
            hexa = ts.read(4)
            aa= bytearray(hexa)
            value=struct.unpack('<f', aa)[0]
            print(value)

"""
Seek can be called one of two ways:
x.seek(offset)
x.seek(offset, starting_point)

starting_point can be 0, 1, or 2
 0 - Default. Offset relative to beginning of file
 1 - Start from the current position in the file
 2 - Start from the end of a file (will require a negative offset)
"""
# import and manage emoncms PHPFINA objects
class PHPFina:
    # the constructor - you just give the number of the timeserie (nb)
    def __init__(self,nb,dir="phpfina"):
        """ initialize the metas """
        #starttime and interval expressed in seconds
        #starttime is a unixtimestamp
        self._startTime = 0
        self._interval = 0
        self._nb = nb
        # the unixtimestamp in seconds at which you decide to start sampling
        self._SamplingPos = 0
        # the period in second at which you sample from the timeseries
        self._step = 3600
        self._dir = dir
        self._datas = []

    # stores and returns the metas
    def getMetas(self):
        with open(self._dir+"/"+str(self._nb)+".meta", "rb") as metas:
            # Seek a specific position in the file and read N bytes
            metas.seek(8, 0)
            interval=readBytes(4,metas)
            metas.seek(12, 0)
            startTime=readBytes(4,metas)
            print("startTime {} ou {}s en unixtimestamp".format(humanDate(startTime),startTime))
            print("interval {}s".format(interval))
            self._startTime=startTime
            self._interval=interval

    def setStart(self,unixTimeStart):
        self._SamplingPos=int((unixTimeStart-self._startTime)/self._interval)
        print("Timeserie {} sampling will start on record number {}".format(self._nb,self._SamplingPos))

    # stores an array of values extracted from the timeserie
    # nbSteps datas are sampled from _SamplingPos at a period equal to _step
    # a PHPFina file is made of 4 bytes float values, with NAN when nothing was recorded from the sensor
    def getDatas(self,nbSteps):
        start = self._SamplingPos
        nbPtInStep=self._step/self._interval
        with open(self._dir+"/"+str(self._nb)+".dat", "rb") as ts:
            for i in range(nbSteps):
                position=int((start+i*nbPtInStep)*4)
                ts.seek(position, 0)
                hexa = ts.read(4)
                aa= bytearray(hexa)
                value=struct.unpack('<f', aa)[0]
                if math.isnan(value):
                    #print("timeseries {} - there is a NAN at position {} in the timeserie or point number {}".format(self._nb,position,i))
                    j=1
                    while True:
                        ramble=position+j*4
                        ts.seek(ramble, 0)
                        hexa = ts.read(4)
                        aa= bytearray(hexa)
                        value=struct.unpack('<f', aa)[0]
                        if math.isnan(value):
                            j+=1
                        else:
                            break
                    #print("we found a value {} {}s after the NAN".format(value,self._interval*j))
                #print(position)
                #print(value)
                #input("press key")
                self._datas.append(value)

    # only for "energy" timeseries
    # accumulates the power for each step and outputs the energy consumption in Kwh within the step to come
    def getKwh(self,nbSteps):
        start = self._SamplingPos
        nbPtInStep=int(self._step/self._interval)
        #print("an hour is sectionned in {} parts".format(nbPtInStep))
        with open(self._dir+"/"+str(self._nb)+".dat", "rb") as ts:
            for i in range(nbSteps):
                position=int((start+i*nbPtInStep)*4)
                acc=0
                ts.seek(position, 0)
                hexa = ts.read(4*nbPtInStep)
                aa= bytearray(hexa)
                values=struct.unpack('<{}f'.format(nbPtInStep), aa)
                for j in range(len(values)):
                    if math.isnan(values[j]):
                        val=0
                    else:
                        val=values[j]
                    acc+=val*self._interval
                if math.isnan(acc):
                    acc=0
                    print("BUG : consumption for full step is NAN !")
                # kwh conversion !!
                self._datas.append(0.001*acc/3600)

    # EmonCMS can provide the accumulated kwh feed over the whole period of the instrumentation, like an energy meter.
    # from the accumulated kwh feed, the kwh consumed per hour can be recalculated
    # NOTA NOTA NOTA : recalculating from the instantaneous power is a better option
    # Indeed, there may be missing data in feeds dedicated to the accumulation of Kwh.
    # Experience shows that only instant power feeds corrected in real time during monitoring.
    def unAcc(self):
        unAccvalues=[]
        for i in range(len(self._datas)-1):
            if self._datas[i+1]-self._datas[i] <0:
                print("bug")
                print(i)
                print("values are {} and {}".format(self._datas[i+1],self._datas[i]))
            unAccvalues.append(self._datas[i+1]-self._datas[i])
        self._datas=unAccvalues

print ("début hiver : {} ou {}s en unixtimestamp".format(humanDate(winterStart),winterStart))

# the indoor temperature recorded
inDoorT=PHPFina(191)
inDoorT.getMetas()
inDoorT.setStart(winterStart)

# the outdoor conditions
outDoorT=PHPFina(1)
outDoorT.getMetas()
outDoorT.setStart(winterStart)

# feed 139 stores instant power dedicated to heating
# we could have used feed 140 which is the kwh accumulated feed
powerW=PHPFina(139)
powerW.getMetas()
powerW.setStart(winterStart)

# we assume the winter period to be something like 200 or 210 days
nbWinterHours=210*24;
inDoorT.getDatas(nbWinterHours)
outDoorT.getDatas(nbWinterHours)
powerW.getKwh(nbWinterHours+1)

meaninDoorT = np.asarray(inDoorT._datas).mean()
stdinDoorT = np.asarray(inDoorT._datas).std()
meanoutDoorT = np.asarray(outDoorT._datas).mean()
stdoutDoorT = np.asarray(outDoorT._datas).std()
print("Tint mean {} std {}".format(meaninDoorT,stdinDoorT))
print("Tout mean {} std {}".format(meanoutDoorT,stdoutDoorT))

plt.subplot(2, 1, 1)
plt.plot(inDoorT._datas)
plt.ylabel("Temp in °C")
plt.plot(outDoorT._datas)
plt.ylabel("In and Out Temp in °C")

plt.subplot(2, 1, 2)
plt.ylabel("Kwh used for heating per step")
plt.plot(powerW._datas)

plt.xlabel("Steps - one step = {}s".format(inDoorT._step))
plt.show()

float_data=np.zeros((len(inDoorT._datas),3))
float_data[:,0]=outDoorT._datas
float_data[:,1]=inDoorT._datas
float_data[:,2]=powerW._datas[0:len(inDoorT._datas)]

#print(float_data[0:504,:])
#print(inDoorT._datas[0:504])
#print(outDoorT._datas[0:504])
#print(powerW._datas[0:504])

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=1):
    if max_index is None:
        max_index=len(data) -delay - 1
    i=min_index+lookback
    datasetNb=0
    while 1:
        datasetNb+=1
        #print("construction of set {} - index in data is {}".format(datasetNb,i))
        # we create an array with the end indexes of each training dataset in the batch
        # we have 128 element in the batch, so the array is of size 128
        # the next element is just shifted by 1 timestep in "data" compared to the previous one.
        # 2 methods : shuffle or random one and chronological
        if shuffle:
            rows = np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            # print(rows)
            i+=len(rows)
        # data.shape[-1] returns the last dimension of data, ie the number of physical features
        # the training dataset structure is (samples,time,features)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        # print("empty samples shape is {}".format(samples.shape))
        targets=np.zeros((len(rows),))
        # print("empty targets shape is {}".format(targets.shape))
        for j, row in enumerate(rows):
            # we sample values in data, over a size range equal to lookback, with a sampling period equal to step
            # we use the range python function for this
            indices = range(rows[j]-lookback, rows[j], step)
            #print("we are at row {} of dataset {} and the sampling range in data is {}".format(j,datasetNb,indices))
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
            #input("press a key")
        yield samples, targets

lookback=12
delay=1
# a step is one hour
step=1
val_steps=168-lookback

train_gen=generator(float_data[5:504+5,:],lookback,delay,0,None)
val_gen=generator(float_data[504+5:504+5+168],lookback,delay,0,None)

for i in range(10):
    samples,targets=next(train_gen)
    plt.figure()
    plt.suptitle("a training set with target {}".format(targets[0]))
    plt.subplot(2, 1, 1)
    plt.plot(samples[0,:,0])
    plt.plot(samples[0,:,1])
    plt.subplot(2, 1, 2)
    plt.hist(samples[0,:,2])
    plt.show()

import tensorflow as tf
print("TF", tf.__version__)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(lookback//step,float_data.shape[-1])))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss='mae')
start_time = time.time()
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

print("Execution time in %s seconds ---" % (time.time() - start_time))

loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title("validation and training loss")
plt.legend()
plt.show()

"""
checkPoints(1,64742,300)
"""
