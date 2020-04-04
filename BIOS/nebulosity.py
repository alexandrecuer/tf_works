import os
import math
import numpy as np
import time
import datetime
from matplotlib import pyplot as plt
import copy
import requests

# this script details how to interrogate the opendata soft API
# https://public.opendatasoft.com or https://data.opendatasoft.com
# this API contains historical datas from Météo France
# the station number is a field named numer_sta
# to search a station number in auvergne rhone alpes
# https://public.opendatasoft.com/api/v2/opendatasoft/datasets/donnees-synop-essentielles-omm%40public/records?where=nom_reg%20like%20%22auvergne%22&rows=10&select=numer_sta%2C%20nom_reg%2C%20libgeo

# we use the requests library which is a port of the classic unix command-line tool curl, used for transferring data from http servers
# the policy is to retrieve in one shot one year of historical data
# here we focus on nebulosity

# it is possible to retrieve only a single measure
# eg: refine.date=2018%2F01%2F01%2F01T00%3A00%3A00%3A00
# the date is url encoded in an hexa fashion
# %2F is /
# %3A is :
# %22 is "
# cf https://www.nicolas-hoffmann.net/utilitaires/codes-hexas-ascii-unicode-utf8-caracteres-usuels.php

server="https://data.opendatasoft.com/explore/dataset"
dataset='donnees-synop-essentielles-omm@public'
# clermont-ferrand station is number 07460
# Lyon/Satolas(Colombier-Saugnieu) is number 07481 for example. It is the nearest station close to grenoble
station="07460"
year=2018
tz="Europe/Paris"

req='{}/{}/download/?format=csv&q=numer_sta%3D%22{}%22&sort=date&refine.date={}'.format(server,dataset,station,year)
req='{}&timezone={}&use_labels_for_header=true&csv_separator=%3B'.format(req,tz)
#print(req)

response = requests.get(req)

data=response.text

# we fix here the presumed timestep in h/s
# for data coming from météo france, timestep is usually 3 hours
step_in_h=3
step_in_s=step_in_h*3600

# remove blank lines
# cf https://www.developpez.net/forums/d821212/autres-langages/python/general-python/reperer-derniere-ligne-d-fichier/
data = data.rstrip('\n\r')

lines = data.split('\n')
header = lines[0].split(';')
lines = lines[1:]
print("we've got {} lines and {} columns".format(len(lines), len(header)))

# fetching the time_stamp and nebulosity index
for i in range(len(header)):
    if header[i].lower()=="date":
        print("timestamp is column {}".format(i))
        ts_index=i
    if header[i].lower()=="nébulosité  des nuages de l' étage inférieur":
        neb_index=i
        print("nebulosity is column {}".format(i))

# raw_data shape is (time,features)
raw_data=np.zeros((len(lines),2))
missing=0
for i,line in enumerate(lines):
    x = line.split(';')
    # removing the last occurence of : in the time string
    # see https://docs.python.org/fr/3.6/library/datetime.html#strftime-strptime-behavior
    # and https://stackoverflow.com/questions/12281975/convert-timestamps-with-offset-to-datetime-obj-using-strptime
    time_str=x[ts_index][::-1].replace(":","",1)[::-1]
    # converting to unixtimestamp
    raw_data[i,0]=time.mktime(datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S%z').timetuple())
    if x[neb_index]:
      raw_data[i,1]=int(x[neb_index])
    else:
      raw_data[i,1]= math.nan
      missing+=1

print("according to the presumed time stamp, we should have {} points".format(365*24/step_in_h))
print("missing datas : {}".format(missing))

#remove eventual lines full of zeros
raw_data = raw_data[~np.all(raw_data == 0, axis=1)]

# reorder by ascending timestep
# https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
# datas are supposed to be already sorted by date but the trick is essential
raw_data=raw_data[raw_data[:,0].argsort()]

float_data=copy.deepcopy(raw_data)
# replace nan values by previous float value
for i in range(float_data.shape[0]):
    if math.isnan(float_data[i,1]) :
        float_data[i,1]=float_data[i-1,1]

# last sanity check is to regularize the timestep as some steps can be missing
full_data=np.zeros((365*24//step_in_h,2))
full_data[0,0]=float_data[0,0]
full_data[0,1]=float_data[0,1]
index=1
for i in range(1,full_data.shape[0],1):
    full_data[i,0]=full_data[i-1,0]+step_in_s
    # if full_data timestep is greater than or equal to the timestep of float_data at index i,
    # we can record a new value in full_data and increment index i
    if full_data[i,0] >= float_data[index,0] and index <= float_data.shape[0]-2:
        full_data[i,1]=float_data[index,1]
        index+=1
    else:
        full_data[i,1]=full_data[i-1,1]
        #print("delta is {} for i {}".format(float_data[index,0]-full_data[i,0],i))

xrange=np.arange(raw_data.shape[0])
ax1=plt.subplot(311)
plt.plot(raw_data[:,1],color="red")
plt.scatter(xrange,raw_data[:,1],color="orange",marker='o',s=4)
ax2=ax1.twinx()
plt.plot(raw_data[:,0],color="blue",label="unixtimestamp")
plt.legend(loc='upper right')
plt.subplot(312,sharex=ax1)
plt.plot(float_data[:,1],color="orange")
ax3=plt.subplot(313)
plt.plot(full_data[:,1],color="orange",label="nebulosity in Octa")
plt.legend(loc='upper left')
ax4=ax3.twinx()
plt.plot(full_data[:,0],color="blue",label="unixtimestamp")
plt.legend(loc='upper right')
plt.show()
