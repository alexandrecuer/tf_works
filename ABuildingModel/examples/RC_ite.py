from src.modelrc import *
import numpy as np

# number of points in an hour
nbptinh=2

dir="../phpfina"

house="ite"
params=[ {"id":1,"name":"outdoor temp","color":"blue","action":"smp"},
         {"id":170,"name": "kitchen","color":"purple","action":"smp"},
         {"id":167,"name": "livingroom","color":"orange","action":"smp"},
         {"id":173,"name":"bathroom","color":"green","action":"smp"},
         {"id":176,"name":"bedroom","color":"#b6e91f","action":"smp"},
         {"id":145,"name":"hvac power (W)","color":"red","action":"smp"},
         {"id":296,"name":"solar power (W)","color":"yellow","action":"smp"}]

# initial guess
# scaling
p0=np.array([1e+6,1e+6,1e-2,1e-2,1e-2])
# weights
w0=np.array([1.0,4.0,1.0,1.0,1.0])

ite=RC_model(house,params,nbptinh,p0,w0,FINASun=True)
ite.buildSet(1547121600,15,[0,5,6],[1,5],dir)
ite.buildSet(1545902400,30,[0,5,6],[1,5],dir)
#for testing
ite.buildSet(1540166400,4,[0,5,6],[1,5],dir)
ite.buildSet(1540166400,200,[0,5,6],[1,5],dir)
guess = np.array([ ite._truth[1][0], ite._truth[1][0]-4 ])
ite.viewSet(2,guess,full=False)
ite.viewSet(3,guess,full=True)
input("press any key")

"""
# dynamic visual approach to evaluate the influence of each parameter
plan=[ [ 20  , 25  , 100 , 25   , 50  ,  50 , 60    ,100   ,30   ],
       [ 1   , 0.5 , 0.1 , 0.01 , 0.01,-0.05,-0.0025,-0.025,-0.25],
       ["go" ,"gof","gof","gof" ,"gof","gof","gof"  ,"gof" ,"gof"],
       [ 0   , 1   , 2   , 3    , 4   , 2   , 4     , 2   , 0    ]
       ]

ite.explorationMatrix(plan,verbose=False)
ite.explore(1,guess)
"""


# scaling
#_p0=np.array([1e+7,1e+7,1e-2,1e-2,1e-2])
# weights
#_w0=np.array([2.0,1.65,6.0,1.25,1.35])
#ite.setWeigths(_p0,_w0)

ite.optimize(1,guess)
ite.viewSet(1,guess,full=False)
ite.viewSet(2,guess,full=False)
ite.viewSet(3,guess,full=False)

"""
_p0=np.array([1e+7,1e+12,1e-3,1e-7,1e-2])
_w0=np.array([1.38,7.08,7.95,4.33,2.26])
ite.setWeigths(_p0,_w0)
guess = np.array([ ite._truth[1][0], ite._truth[1][0] ])
ite.viewSet(3,guess,full=False)
"""
