import numpy as np
import random
import math
#import tensorflow as tf
from datetime import datetime
import struct
import matplotlib.pylab as plt
#import time
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

## 10 minutes approach
Hslices=6
nbsteps=210*24*Hslices
step=step_h//Hslices
winter=GoToTensor(params,step,winterStart,nbsteps)
input("press a key")


t = np.arange(0, nbsteps, Hslices)
#print("last element in t : {}".format(t[-1]))

plt.subplot(221)
#Out temperature
plt.plot(winter[:,0])
plt.plot(t,winterh[:,0])
#In temperature
plt.plot(winter[:,1])
plt.plot(t,winterh[:,1])
plt.ylabel("In and Out Temp in °C")

plt.subplot(223)
plt.ylabel("Kwh used for heating per step")
plt.plot(winter[:,2])
plt.plot(t,winterh[:nbHsteps,2])
plt.xlabel("Steps - one step = {}s".format(step))

plt.subplot(224)
reag=np.zeros((nbHsteps))
i=j=0
while i < nbsteps:
    reag[j]=winter[i:i+Hslices,2].sum()
    i+=Hslices
    j+=1
plt.plot(t,reag)
plt.ylabel("Kwh({}s)->Kwh({}s))".format(step,step_h))

ChecknR(winter,True)
plt.subplot(222)
plt.plot(winter[:,0])
plt.plot(winter[:,1])
plt.plot(winter[:,2])
plt.ylabel("After regul.")

plt.show()
