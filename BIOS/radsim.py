from sunModel import globalSunRadiation
import numpy as np
import matplotlib.pylab as plt
import time
import struct
from PHPFina import PHPFina, newPHPFina

def InitializeFeed(nb,step,start):
    feed=PHPFina(nb,step)
    feed.getMetas()
    feed.setStart(start)
    return feed

# given some PHPFina feeds, a period and a start (as a unix timestamp) both in seconds
def GoToTensor(params,step,start,nbsteps):
    #print("going to tensor for {} feeds".format(len(params)))
    float_data=np.zeros((nbsteps,len(params)+1))
    for i in range(len(params)):
        #print("feed number {}".format(i))
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

def strForODS(ts):
    """
    format a unixtimestamp into a UTC aware timestring formatted for opendatasoft API
    """
    tt=time.gmtime(ts)
    str=time.strftime('%Y-%m-%dT%H:%M:%S%z',tt)
    return "{}{}:{}".format(str[:-4],str[-4:-2],str[-2:])

# Escurolles is lat 46°8'39'' long 3°15'58''
# in decimal deg.
lat=46.1441667
# in decimal deg.
long=3.2661111
# in meter
alt=300

smpStart=1514764800+7600
tDays=365*2

nbptinh=12

sun=globalSunRadiation(lat,long,alt,nbptinh,smpStart,tDays)
"""
for i in [1,100,200,300]:
    print("day  duration of day {} is {} h".format(i, sun.sunDuration(i)))
"""
sun.generate()

ax1=plt.subplot(211)
plt.plot(sun._datas[:,0],label="global radiation in W/m2",color="orange")
plt.legend(loc='upper left')
ax2 = ax1.twinx()
plt.plot(sun._datas[:,1],label="Linke trouble",color="green")
plt.legend(loc='upper right')
plt.subplot(212,sharex=ax1)
plt.plot(sun._datas[:,2],label="gamma-height",color="orange")
plt.plot(sun._datas[:,3],label="alpha-zenith",color="red")
plt.plot(sun._datas[:,4],label="omega-hour angle",color="pink")
plt.legend(loc='upper right')
plt.xlabel("Time - step=h/{}".format(nbptinh))
plt.show()

ts=sun._utsStart
print(ts)
ts_end=ts+tDays*3600*24
print(ts_end)

from openData import openData
dataset='donnees-synop-essentielles-omm%40public'
station="07460"
start=strForODS(ts)
stop=strForODS(ts_end)
tz="UTC"
fields=["date","nbas","tc"]
# we fix here the presumed timestep in hour
# for data coming from météo france, timestep is usually 3 hours
step_in_h=3

synop=openData(dataset,station,start,stop,fields,tz,step_in_h,year=False)

synop.retrieve()

# cloud attenuation factor
indice=0
Kct=np.zeros(sun._datas.shape[0])
for i in range(sun._datas.shape[0]):
    if i % (nbptinh*step_in_h) == 0:
        Kc=(1-0.75*(synop._full_data[indice,1]/9)**3.4)
        if indice < synop._full_data.shape[0]:
            indice+=1
    Kct[i]=Kc

sun._datas[:,0]=Kct*sun._datas[:,0]

sun.energy()
print(sun._E)

nb=296
step=3600//nbptinh
newPHPFina(nb,ts,step,sun._datas[:,0])

params=[ {"id":nb,"name":"sun radiation","color":"yellow","action":"smp"} ]
nbsteps=tDays*24*nbptinh
feed=GoToTensor(params,step,ts,nbsteps)

plt.subplot(111)
plt.plot(feed[:,0])
plt.show()
