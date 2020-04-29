from src.modelrc import *
from src.tools import PHPFina,getMetas
import numpy as np
import math

# number of points in an hour
nbptinh=2
step=3600//nbptinh

dir="../labo/phpfina"

house="labo cellules"
params=[ {"id":18,"name":"outdoor temp","color":"blue","action":"smp"},
         {"id":56,"name": "cells indoor temp","color":"purple","action":"smp"},
         {"id":66,"name":"cells hvac power (W)","color":"red","action":"smp"},
         {"id":21,"name":"solar power (W)","color":"yellow","action":"smp"}]

# heated floor area in m2 ? maybe 1000 m2 as the building is 3000 m2 and has got 3 heating circuits
floor=1000
# building volume in m3
vb=floor*3
# air bulk density in kg/m3
rho_air=1.22
# air heat capacity in J/(kg.K)
c_air=1004
# res in J/K
cres=c_air*rho_air*vb

print("cres should be {}".format("{:e}".format(cres)))
# initial guess
# scaling
p0=np.array([1e+6,1e+6,1e-4,1e-4,1e-4])
# weights
w0=np.array([4.0,4.0,1.0,1.0,1.0])

getMetas(18,dir=dir)
getMetas(56,dir=dir)
getMetas(66,dir=dir)
getMetas(21,dir=dir)

start=1577785800
tDays=30

labocells=RC_model(house,params,nbptinh,p0,w0,FINASun=True)

# just to verify the sampling positions....
for i in range(len(params)):
    id=params[i]["id"]
    feed=PHPFina(id,step,dir)
    feed.getMetas()
    feed.setStart(start)
    nbsteps=tDays*24*nbptinh
    feed.getDatas(nbsteps)
    print("feed {} will be read at position {}".format(id,feed._SamplingPos))


labocells.buildSet(start,tDays,[0,2,3],1,dir)

"""
how to make a guess for the first initial envelope temperature ?
i is the set number in the RC_model
"""
def f(i):
    x=labocells._truth[i][0]
    # external temperature
    y=labocells._inputs[i][0,0]
    return (x+2*y)/3


guess = np.array([ labocells._truth[0][0], f(0) ])
labocells.viewSet(0,guess,full=False)

#increasing sun
K=50
labocells._inputs[0][:,2]=K*labocells._inputs[0][:,2]
labocells.viewSet(0,guess,full=False)

input("press any key to launch the optimization")

labocells.optimize(0,guess,verbose=False)
labocells.viewSet(0,guess,full=False)

"""
input("press any key to launch a second optimization")
p0=np.array([1e+9,1e+7,1e-5,1e-2,1e-4])
w0=np.array([1.0,1.0,1.0,1.0,1.0])
labocells.setWeigths(p0,w0)
labocells.optimize(0,guess)
labocells.viewSet(0,guess,full=False)
"""
"""
input("press any key to launch a third optimization")
p0=np.array([1e+6,1e+6,1e-5,1e-2,1e-4])
w0=np.array([1.0,1.0,1.0,1.0,1.0])
labocells.setWeigths(p0,w0)
labocells.optimize(0,guess)
labocells.viewSet(0,guess,full=False)
"""

# adding set 1 to check if things can be correctly predicted...
labocells.buildSet(1579492800,30,[0,2,3],1,dir)
guess = np.array([ labocells._truth[1][0], f(1) ])
labocells._inputs[1][:,2]=K*labocells._inputs[1][:,2]
labocells.viewSet(1,guess,full=False)

# adding set 2 to check if things can be correctly predicted...
labocells.buildSet(1584252000,20,[0,2,3],1,dir)
guess = np.array([ labocells._truth[2][0], f(2) ])
labocells._inputs[2][:,2]=K*labocells._inputs[2][:,2]
labocells.viewSet(2,guess,full=False)
