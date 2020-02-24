import os
import numpy as np
import random
import time
import struct
import matplotlib.pylab as plt
from PHPFina import PHPFina,humanDate
from pandas import DataFrame
from sklearn import linear_model
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

winterStart=1539950400
nbptinh=1
nbsteps=210*24*nbptinh
step=3600//nbptinh
params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
winter=GoToTensor(params,step,winterStart,nbsteps)

history_size=10
# how many hours in the future are we going to simulate ?
goto=48
# where to cut the dataset
keep_for_validation=1500

debug=False
# l is the number of datasets we can construct with the timeseries
# we assume to make an only prediction 1 hour in the future, so we need to keep the last point aside
l=winter.shape[0]-1-history_size
# tsize is the training size
tsize=l-keep_for_validation

print("size for multiregression is : {}".format(tsize))
train_mean = winter[:history_size+tsize].mean(axis=0)
train_std = winter[:history_size+tsize].std(axis=0)
print(train_mean)
print(train_std)
# regularize all the dataset but keep a copy in order not to recalculate things for nothing
physics = copy.deepcopy(winter)
winter = (winter-train_mean)/train_std

# we build the matrix column by column
msize=winter.shape[0]-history_size
float_data=np.zeros((msize,winter.shape[1]*history_size))
indice=0
for j in range(winter.shape[1]):
    for i in range(history_size):
        float_data[:,indice]=winter[i:i+msize,j]
        indice+=1
print(float_data.shape)
futureTint=winter[history_size:winter.shape[0],1]
print(len(futureTint))

regr = linear_model.LinearRegression()
regr.fit(float_data[0:tsize], futureTint[0:tsize])

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

plt.subplot(111)
plt.plot(regr.coef_)
plt.show()

lab = ["out. temp","indoor temp", "Kwh"]
col = ["blue", "green", "red"]

history_range=list(range(-history_size, 0))
future_range=list(range(0,goto))

for nbset in range(tsize,float_data.shape[0]-goto-1):
#for z in range(10):
#    nbset=random.randint(0,float_data.shape[0]-goto)

    ax1 = plt.subplot(311)
    plt.ylabel("T in and out °C")
    plt.title("dataset {}".format(history_size+nbset))

    plt.xlim([history_range[0], goto+5])

    plt.plot(future_range,futureTint[nbset:nbset+goto], 'rx', label = "True future", color=col[1])

    # pred is an array to store the predictions step by step
    pred=[]
    for k in range(goto):
        # the input sample for the model
        # in each sample, we find the indoor T values from index history_size to 2*history_size-1
        # the number of indoor T values is history_size
        # l is the size of the pred array
        # to make the predictions step by step :
        # - if l < history_size, we have to replace the last l T values by the predicted ones
        # - if l >= history_size, we have to replace the whole history_size values in the sample by the predicted ones
        sample=copy.deepcopy(float_data[nbset+k])
        if debug:
            print(pred)
            print("before injection")
            print(sample)
        if len(pred)>0:
            if len(pred)>history_size:
                simTs=pred[-history_size:]
            else:
                simTs=pred
            sample[2*history_size-min(len(pred),history_size):2*history_size]=simTs
        if debug:
            print("after injection")
            print(sample)
            input("press any key")
        prediction=np.sum(regr.coef_*sample)+regr.intercept_
        pred.append(prediction)

    plt.plot(future_range,pred,'o',label="multilinear prediction", color=col[1])
    for i in range(2):
        plt.plot(history_range, float_data[nbset,i*history_size:(i+1)*history_size], 'rx', color=col[i])
        plt.plot(history_range,winter[nbset:nbset+history_size,i],color=col[i], label=lab[i])
        plt.plot(future_range,winter[nbset+history_size:nbset+history_size+goto,i], color=col[i])
    plt.legend()

    plt.subplot(312, sharex=ax1)
    plt.plot(future_range,train_mean[i]+train_std[i]*np.array(pred),'o',label="multilinear prediction", color=col[1])
    for i in range(2):
        plt.plot(history_range, train_mean[i]+train_std[i]*float_data[nbset,i*history_size:(i+1)*history_size], 'rx', color=col[i])
        plt.plot(history_range,physics[nbset:nbset+history_size,i],color=col[i], label=lab[i])
        plt.plot(future_range,physics[nbset+history_size:nbset+history_size+goto,i], color=col[i])
    plt.legend()

    plt.subplot(313, sharex=ax1)
    plt.xlim([history_range[0], goto+5])
    plt.plot(history_range, train_mean[2]+train_std[2]*float_data[nbset,2*history_size:3*history_size], 'rx', color=col[2] )
    plt.plot(history_range,physics[nbset:nbset+history_size,2],color=col[2], label=lab[2])
    plt.plot(future_range,physics[nbset+history_size:nbset+history_size+goto,2], color=col[2])
    plt.legend()

    plt.show()

