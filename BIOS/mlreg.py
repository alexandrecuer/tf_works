import os
import numpy as np
import random
import time
import struct
import matplotlib.pylab as plt
from PHPFina import PHPFina,humanDate
from pandas import DataFrame
from sklearn import linear_model

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

# we just fetch a 10 minutes discretisation of the PHPFina indoor temperature feed
tint10m=GoToTensor([{"id":191,"action":"smp"}],600,winterStart,nbsteps*6)

history_size=24
future=1
# how many hours in the future are we going to simulate ?
goto=100

l=winter.shape[0]-history_size
float_data=np.zeros((l,winter.shape[1]*history_size))
indice=0
for j in range(winter.shape[1]):
    for i in range(history_size):
        float_data[:,indice]=winter[i:i+l,j]
        indice+=1

futureTint=winter[history_size+future:l+future,1]

regr = linear_model.LinearRegression()
regr.fit(float_data[0:futureTint.shape[0]], futureTint)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

plt.subplot(111)
plt.plot(regr.coef_)
plt.show()

history_range=list(range(-history_size+1, 1))

for z in range(10):
    nbset=random.randint(0,float_data.shape[0]-goto)

    plt.subplot(311)
    plt.ylabel("T in and out")
    plt.title("dataset {}".format(nbset))
    #plt.xlim([history_range[0], (future+5)*2])

    plt.xlim([history_range[0], goto+5])
    future_range=list(range(future,future+goto))
    if goto < 20:
        plt.plot(future_range,futureTint[nbset:nbset+goto], 'go', label = "True future")
    elif goto >= 20:
        plt.plot(future_range,futureTint[nbset:nbset+goto], label = "True future")

    # pred is an array to store the predictions step by step
    pred=[]
    for k in range(goto):
        # the input for the model
        # in each input, we find the indoor T values from index history_size to 2*history_size-1
        # the number of indoor T values is history_size
        # l is the size of the pred array
        # to make the predictions step by step, we need to modify the input vector :
        # - if l < history_size, we have to replace the last l indoor T values by the predicted ones
        # - if l >= history_size, we have to replace the whole history_size indoor T values by the predicted ones
        input=float_data[nbset+k]
        if len(pred)>0:
            if len(pred)>history_size:
                simTs=pred[len(pred)-1-history_size:len(pred)-1]
            else:
                simTs=pred
            input[2*history_size-1-min(len(pred),history_size):2*history_size-1]=simTs
        prediction=np.sum(regr.coef_*input)+regr.intercept_
        pred.append(prediction)
        #print(pred)
    if goto < 20:
        plt.plot(future_range,pred,'rx',label="multilinear prediction")
    elif goto >= 20:
        plt.plot(future_range,pred,label="multilinear prediction")
    plt.legend()
    for i in range(winter.shape[1]):
        if i == winter.shape[1]-1:
            plt.subplot(312)
            plt.xlim([history_range[0], goto+5])
            plt.ylabel("Kwh used for heating per step")
        plt.plot(history_range, float_data[nbset,i*history_size:(i+1)*history_size])
    # this is the last subplot with a full range view of indoor and outdoor temperatures
    plt.subplot(313)
    t = np.arange(0, nbsteps*6, 6)
    plt.plot(t,winter[:,0])
    plt.plot(t,winter[:,1])
    plt.plot(tint10m[:,0])
    plt.show()
