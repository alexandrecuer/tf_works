import os
import numpy as np
import time

# the Max Planck weather dataset
# 6 months datasets are available here : https://www.bgc-jena.mpg.de/wetter/weather_data.html
# a bigger dataset can be downloaded via wget https://s3.amazonaws.com/keras_datasets/jena_climate_2009_2016.csv.zip
fname = "jena_climate_2009_2016.csv"
f = open (fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)

# float_data shape is (time,features)
float_data=np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    if i==0:
        print(line.split(',')[0:])
        print(line)
        print(values)
    float_data[i,:]=values

#print (float_data[0,:])
print("data shape is {}".format(float_data.shape))

from matplotlib import pyplot as plt

temp =float_data[:,1]
plt.plot(range(len(temp)),temp)
plt.show()


mean = float_data[:200000].mean(axis=0)
print("the mean is {}".format(mean))
float_data -= mean
std = float_data[:200000].std(axis=0)
print("the standart deviation is {}".format(std))
float_data /= std


# lookback : how many timesteps in the past the input data should go
# delay : how many timesteps in the future the target should be
# we create a generator with yield in order to generate the batches on the fly
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
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


# ***********************************************************************************************************
# ***********************************STRUCTURE OF A DATASET *************************************************
# ***********************************************************************************************************
# a single training dataset is made of the 14 weather features, 10 days in the past
# its targets is a single temperature value, 1 day in the future
# as we are going to sample each 6 steps=one hour, we will have 1440/6=240 samples per feature in the dataset
# the shape of the dataset will be (240,14)
# in the generator, samples[j] is the "single" dataset
# ***********************************************************************************************************
# we are going to train the NN on batches including 128 datasets ********************************************
# ***********************************************************************************************************
lookback=1440
delay=144
# a step is one hour
step=6
# we create the generators
train_gen=generator(float_data,lookback,delay,0,200000,True)
val_gen=generator(float_data,lookback,delay,200001,300000)
test_gen=generator(float_data,lookback,delay,300001,None)

val_steps=(300000-200001-lookback)//128



def evaluate_naive_method():
    batch_maes=[]
    for step in range(val_steps):
        samples,targets=next(val_gen)
        #print("we are at step {}".format(step))
        # samples is an array of matrix
        # in each matrix of samples, we extract the last element of column 1, which is temperature
        preds=samples[:,-1,1]
        #print(preds)
        #input("press any key")
        mae=np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

start_time = time.time()
#evaluate_naive_method()

import tensorflow as tf
print("TF", tf.__version__)
#import keras as keras
#from keras.models import sequential
#from keras import layers
#from keras.optimizers import RMSprop
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(lookback//step,float_data.shape[-1])))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss='mae')
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
