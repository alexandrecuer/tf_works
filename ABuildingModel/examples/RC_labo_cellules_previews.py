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

# initial guess
# scaling
p0=np.array([1e+6,1e+6,1e-4,1e-4,1e-4])
# weights
w0=np.array([4.0,4.0,1.0,1.0,1.0])

start=1577785800
tDays=30

labocells=RC_model(house,params,nbptinh,p0,w0,FINASun=True)

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


# dynamic visual approach to evaluate the influence of each parameter
# we only play on the weights, not on the scales
# first we increase Rf by small steps of 0.1 till we've added 2 (20 steps) : we should go from 1e-4 to 3e-4
# second we increase Cres by big steps of 10 till we've added 400 (40 steps) : we should go from 4e6 to 404e6, ie 4.04e8
# third we increase Cs by big steps of 100 till we've added 2000 (20 steps) : we should go from 4e6 to 2004e6, ie 2e9
plan=[ [ 20  ,40   ,20   ],
       [ 0.25 ,10   ,100  ],
       ["gof","gof","gof"],
       [ 4   ,0    ,1    ]
       ]

labocells.explorationMatrix(plan,verbose=False)
labocells.explore(0,guess)

# scaling
p0=np.array([1e+8,1e+9,1e-4,1e-4,1e-4])
# weights
w0=np.array([4.0,2.0,1.0,1.0,3.0])

labocells.setWeigths(p0,w0)
labocells.viewSet(0,guess,full=False)
