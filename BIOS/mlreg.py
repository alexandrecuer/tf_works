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

history_size=48
future=12

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

history_range=list(range(-history_size, 0))

for z in range(10):
    nbset=random.randint(0,float_data.shape[0])

    plt.subplot(211)
    plt.ylabel("T in and out")
    plt.title("dataset {}".format(nbset))
    plt.xlim([history_range[0], (future+5)*2])
    plt.plot(future,futureTint[nbset],'go', label="True future")
    prediction=np.sum(regr.coef_*float_data[nbset])+regr.intercept_
    plt.plot(future,prediction,'rx', label="multilinear prediction")
    plt.legend()
    for i in range(winter.shape[1]):
        if i == winter.shape[1]-1:
            plt.subplot(212)
            plt.xlim([history_range[0], (future+5)*2])
            plt.ylabel("Kwh used for heating per step")
        plt.plot(history_range, float_data[nbset,i*history_size:(i+1)*history_size])

    plt.show()
