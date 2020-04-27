from src.tools import *
import numpy as np
import matplotlib.pylab as plt
import time

"""
simulates 2 years of sun for a point on the earth defined by its geographical coordinates (lat, long, alt)
the simulated radiation is for an open sky, without any cloud
connects to opendata and retrieves nebulosity on the same period, given the id of the closest meteofrance weather station
modulates the simulated radiation, given the nebulosity datas
save the datas to a PHPFina feed in a specified dir
"""

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

if (synop._uts==ts):
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

    dir="../phpfina"
    # find the feed with the highest number
    import os
    files=os.listdir(dir)
    feeds=np.zeros(len(files))
    for i,file in enumerate(files):
        if file.endswith(".dat"):
            feeds[i]=file.split(".dat")[0]
    # by adding 1 we are sure to create a new feed
    nb=int(np.max(feeds))+1

    step=3600//nbptinh
    newPHPFina(nb,ts,step,sun._datas[:,0],dir)

    nbsteps=tDays*24*nbptinh

    feed=PHPFina(nb,step,dir)
    feed.getMetas()
    feed.setStart(ts)
    feed.getDatas(nbsteps)

    plt.subplot(111)
    plt.plot(feed._datas)
    plt.show()
else:
    print("cannot synchronise synthetic sun with clear sky and nebulosity feed")
