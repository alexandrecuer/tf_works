import numpy as np
import random
import math
import time
import struct
import matplotlib.pylab as plt
from PHPFina import PHPFina,humanDate
import tensorflow as tf
import sys
import copy

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
        else:
            return False
    return float_data

# if target_size is 1, prediction is the next point
# if target_size > 1, there is target_size points in the prediction
def multivariate_data(dataset, target, rows, history_size, target_size):
  data = []
  labels = []
  for j in range(len(rows)):
      indices = range(rows[j]-history_size, rows[j])
      data.append(dataset[indices])
      labels.append(target[rows[j]:rows[j]+target_size])
      """
      if target_size == 1:
          labels.append(target[rows[j]])
      elif target_size > 1:
          labels.append(target[rows[j]:rows[j]+target_size])
      """
  return np.array(data), np.array(labels)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(loss))
  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.show()

#***********************************************************************************
#***********************************************************************************
winterStart=1539950400
nbptinh=1
nbsteps=210*24*nbptinh
step=3600//nbptinh
params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
winter=GoToTensor(params,step,winterStart,nbsteps)
print(winter.shape)

history_size=10
# we can use target_size to make a multi-size prediction
# when modifying target_size or history_size, the network has to be retrained !!!!
target_size=1
# how many hours in the future are we going to simulate ?
# if singlesim is set to true, a single simulation is done and the goto param is not used
goto=48
singlesim=False
# where to cut the dataset
keep_for_validation=1500

debug=False

training=True

# we cannot mix multi-simulation and muti-size prediction
if singlesim==False and target_size>1:
    print("cannot go further - in multisim mode, you need to fix target_size=1")
    sys.exit()
#***********************************************************************************
#***********************************************************************************

# l is the number of datasets we can construct with the timeseries
l=winter.shape[0]-target_size-history_size
# tsize is the training size
tsize=l-keep_for_validation

rows_train=np.arange(history_size,history_size+tsize,1)
rows_val=np.arange(history_size+tsize,winter.shape[0]-target_size,1)
if debug:
    print(rows_train.shape)
    print(rows_val.shape)
    input("press any key")
# mean and std on the training datas
train_mean = winter[:history_size+tsize].mean(axis=0)
train_std = winter[:history_size+tsize].std(axis=0)
print(train_mean)
print(train_std)
# regularize all the dataset but keep a copy in order not to recalculate things for nothing
physics = winter
winter = (winter-train_mean)/train_std
#physics = winter*train_std+train_mean
if training:
    # shuffling the training sets
    np.random.shuffle(rows_train)
    print(rows_train)
    datas_train, labels_train=multivariate_data(winter, winter[:,1], rows_train, history_size, target_size)
    datas_val, labels_val=multivariate_data(winter, winter[:,1], rows_val, history_size, target_size)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(32,input_shape=datas_train.shape[-2:]))
    model.add(tf.keras.layers.Dense(target_size))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
    history = model.fit(datas_train, labels_train, epochs=20,batch_size=50,
                                            validation_data=(datas_val, labels_val))

    plot_train_history(history,'Single Step Training and validation loss')
    model.save('NN_LSTM.h5')
else:
    model = tf.keras.models.load_model('NN_LSTM.h5')

model.summary()

history_range=list(range(-history_size, 0))
if singlesim :
    target_range=list(range(0,target_size))

# uncomment next line if you want to work on a large package including also the training datas
#start=history_size
start=history_size+tsize
print("start is {}".format(start))
stop=winter.shape[0]-target_size
if not singlesim:
    stop=stop-goto
rows=np.arange(start,stop,1)
print("we are going to play on a package of {} datasets".format(len(rows)))
print(rows)
datas, labels=multivariate_data(winter, winter[:,1], rows, history_size, target_size)

lab = ["out. temp","indoor temp", "Kwh"]
col = ["blue", "green", "red"]

for nbset in range(len(labels)):
#for z in range(10):
    # randint(a,b) returns an int N such as a <= N <=b
#    nbset=random.randint(0,len(labels)-1)
    if singlesim:
        end=target_size+5
    else:
        end=goto+5

    ax1=plt.subplot(311)
    plt.title("sample nb {}".format(nbset+start))
    plt.xlim([history_range[0], end])
    # printing the initial dataset (nbset) with crosses
    plt.plot(history_range,datas[nbset][:,0], 'rx', color=col[0])
    plt.plot(history_range,datas[nbset][:,1], 'rx', color=col[1])
    plt.plot(history_range,datas[nbset][:,2], 'rx', color=col[2])

    if singlesim :
        # in singlesim mode, we can do single-size OR multi-size prediction
        # in both case, the output is prediction[0,:]
        plt.plot(target_range,labels[nbset],'rx')
        sample=datas[nbset].reshape(1,history_size,datas.shape[-2:][1])
        prediction=model.predict(sample)
        plt.plot(target_range,prediction[0,:],'o', label="pred.", color=col[1])
    else:
        # pred and truth are arrays to store predictions and truths step by step
        # the oldest data in time is pred[0], the newest data is the last one, ie pred[-1]
        # pred[-4:] returns the last 4 elements of pred, ie the 4 more recent predictions
        # same with sample
        pred=[]
        truth=[]
        for k in range(goto):
            # the input for the model
            position=nbset+k
            # make a deepcopy not to affect datas
            # cf http://lyceeomar.atspace.cc/Copiesup_profonde1.html
            sample=copy.deepcopy(datas[position])
            if debug:
                print("pred length : {}".format(len(pred)))
                print(pred)
            if len(pred)>0:
                if len(pred)>history_size:
                    # we want the last history_size predictions
                    simTs=pred[-history_size:]
                else:
                    simTs=pred
                if debug:
                    print("before injection")
                    print(sample)
                # we have to update the last min(len(pred),history_size) elements in the temperature column (1)
                sample[-min(len(pred),history_size):,1]=simTs
            if debug:
                print("after sim injection")
                print(sample)
                input("press any key")
            sample=sample.reshape(1,history_size,datas.shape[-2:][1])
            prediction=model.predict(sample)
            pred.append(prediction[0,0])
            truth.append(labels[position])
        plt.plot(range(goto),pred,'o',label="pred.", color=col[1])
        plt.plot(range(goto),truth,'rx',label="truths", color=col[1])

    for k in range(3):
        plt.plot(history_range,winter[nbset+start-history_size:nbset+start,k], color=col[k], label=lab[k])
        plt.plot(range(end),winter[nbset+start:nbset+start+end,k], color=col[k])
    plt.legend()

    plt.subplot(312, sharex=ax1)
    for k in range(2):
        plt.plot(history_range,physics[nbset+start-history_size:nbset+start,k], color=col[k], label=lab[k])
        plt.plot(range(end),physics[nbset+start:nbset+start+end,k], color=col[k])
    if singlesim :
        plt.plot(target_range,prediction[0,:]*train_std[1]+train_mean[1],'o',label="pred.", color=col[1])
    else:
        physical_pred=np.array(pred)*train_std[1]+train_mean[1]
        plt.plot(range(goto),physical_pred,'o',label="pred.", color=col[1])
    plt.legend()

    plt.subplot(313, sharex=ax1)
    plt.plot(history_range,physics[nbset+start-history_size:nbset+start,2], color=col[2], label=lab[2])
    plt.plot(range(end),physics[nbset+start:nbset+start+end,2], color=col[2])
    plt.legend()

    plt.show()
