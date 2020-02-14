import numpy as np
import random
import math
#import tensorflow as tf
from datetime import datetime
import time
import struct
import matplotlib.pylab as plt
from PHPFina import PHPFina,humanDate

winterStart=1539950400
print ("début hiver : {} ou {}s en unixtimestamp".format(humanDate(winterStart),winterStart))

def InitializeFeed(nb,step,start):
    feed=PHPFina(nb,step)
    feed.getMetas()
    feed.setStart(start)
    return feed

# given some PHPFina feeds, a period and a start (as a unix timestamp) both in seconds
def GoToTensor(params,step,start,nbsteps):
    print("going to tensor for {} feeds".format(len(params)))
    float_data=np.zeros((nbsteps,len(params)))
    for i in range(len(params)):
        print("feed number {}".format(i))
        feed=InitializeFeed(params[i]["id"],step,start)
        if params[i]["action"]=="smp":
            feed.getDatas(nbsteps)
        elif params[i]["action"]=="acc":
            feed.getKwh(nbsteps)
        if len(feed._datas):
            float_data[:,i]=feed._datas[0:nbsteps]
    return float_data

def ChecknR(tensor,regularize=True):
    mean=tensor.mean(axis=0)
    if regularize:
        tensor-=mean
    std=tensor.std(axis=0)
    if regularize:
        tensor/=std
    print(mean)
    print(std)

# we assume the winter period to be something like 200 or 210 days
# outdoor, indoor, instant power
# instead of feed 139, we could have used feed 140 which is the kwh accumulated feed
params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]

## HOURLY APPROACH
nbHsteps=210*24
step_h=3600
winterh=GoToTensor(params,step_h,winterStart,nbHsteps)

plt.subplot(211)
#Out temperature
plt.plot(winterh[:,0])
#In temperature
plt.plot(winterh[:,1])
plt.ylabel("In and Out Temp in °C")
plt.subplot(212)
plt.ylabel("Kwh used for heating per step")
plt.plot(winterh[:,2])
plt.xlabel("Steps - one step = {}s".format(step_h))
plt.show()

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
        # print("batch nb {} startingpos in data {}".format(datasetNb,i))
        # we create an array with the end indexes of each training dataset in the batch
        # 2 methods : shuffle = randomize VS chronological
        # in chronological mode, element n+1 is just shifted by 1 timestep further in "data" compared to element n
        if shuffle:
            rows = np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
        # data.shape[-1] returns the last dimension of data, ie the number of physical features
        # the training tensor is an array of datasets - shape =  (samples,time,features)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        # print("empty samples shape is {}".format(samples.shape))
        targets=np.zeros((len(rows),))
        # print("empty targets shape is {}".format(targets.shape))
        for j, row in enumerate(rows):
            # we sample values in data, over a size range equal to lookback, with a sampling period equal to step
            indices = range(rows[j]-lookback, rows[j], step)
            #print("Element {} of dataset {} - sampling range in data is {}".format(j,datasetNb,indices))
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
            #input("press a key")
        yield samples, targets

train=3*7*24
val=7*24
lookback=24
delay=4

val_steps=val-lookback
float_data=winterh
train_gen=generator(float_data[0:train,:],lookback,delay,0,None)
val_gen=generator(float_data[train:train+val],lookback,delay,0,None)

"""
for k in range(10):
    samples,targets=next(train_gen)

    for i in range(10):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('time (h)')
        ax1.set_ylabel('T')
        ax1.plot(samples[i,:,0], color='tab:red')
        ax1.plot(samples[i,:,1], color='tab:blue')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Kwh/h')
        ax2.plot(samples[i,:,2], color='tab:green')
        ax2.tick_params(axis='y')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.suptitle("set {} el. {} with target {}".format(k,i,targets[i]))
        plt.show()
"""


import tensorflow as tf
print("TF", tf.__version__)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(lookback,float_data.shape[-1])))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
#model.add(tf.keras.layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
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
plt.title("validation and training loss for lookback {}".format(lookback))
plt.legend()
plt.show()

"""
checkPoints(1,64742,300)
"""
