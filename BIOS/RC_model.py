import numpy as np
from PHPFina import PHPFina,humanDate
import matplotlib.pylab as plt
from datetime import datetime
import random
import math
import copy

def generateSunDay(qs_max, nb):
    """
    generate one day of sun given a qs_max power

    :qs_max: maximum power in W that the sun can deliver during the day
    :nb: number of points in a hour (determine the precision of the discretisation)
    :return: numpy vector of length 24*nb
    """
    # number of steps in a day
    day=24*nb
    sun=np.zeros(day)
    sunrise=8*nb
    sunset=17*nb
    # number of steps of sun in a day
    sunstp=9*nb
    # number of steps after sunrise to achieve zenith
    zenith=4*nb
    for i in range(day):
        if i < sunrise or i > sunset:
            sun[i]=random.random()
        else:
            sun[i]=max(0,qs_max*(2*math.exp(-10*((i-sunrise-zenith)/sunstp)**2)))
    return sun

def generateSunRange(qs_max, nb, size, offset):
    """
    generate sun power over a full sampling period

    :qs_max: maximum power in W that the sun can deliver during the day
    :nb: number of points in a hour (determine the precision of the discretisation)
    :size: number of points of the synthetized sun sample
    :offset: hour of the day to start with the synthesis
    :return: numpy vector of length size
    """
    #print("offset is {}".format(offset))
    sunrange=np.zeros(size)
    sunrange[0:(24-offset)*nb]=generateSunDay(qs_max,nb)[offset*nb:24*nb]
    written=(24-offset)*nb
    while written < size :
        #print("we wrote {} points on a total of {}".format(written,size))
        if written+24*nb > size:
            break
        sunrange[written:written+24*nb]=generateSunDay(qs_max,nb)
        written+=24*nb
    #print("loop finished, we wrote {} on a total of {}".format(written,size))
    if offset > 0:
        sunrange[-offset*nb:]=generateSunDay(qs_max,nb)[0:offset*nb]
    return sunrange

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

def visualize(sample,meta,lib,**kwargs):
    """
    visualization tool

    :sample: tensor with the datas recorded from the field - one column = one monitored parameter
             the 2 last colums are hvac power and sun power in W
             all other columns are temperature fields
    :meta: metadatas list - the minimal infos are name and color, eg [{"name":"first curve","color":"green"},....]
    :lib: title of the graph

    you can add 4 numpy vectors to plot results from simulation
    """
    xrange=np.arange(sample.shape[0])
    ax1=plt.subplot(111)
    ax1.set_ylabel("°C")
    plt.title(lib)
    plt.plot(sample[:,0],label=meta[0]["name"], color=meta[0]["color"])
    for j in range(1,sample.shape[1]-2,1):
        plt.plot(sample[:,j],label=meta[j]["name"], color=meta[j]["color"])
    if len(kwargs):
        # you can plot 4 extra curves with prediction datas
        icons=[':','--','o','*']
        indice=0
        for key, vals in kwargs.items():
            plt.plot(vals,icons[indice],markersize=2, label="{}.".format(key))
            indice+=1
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.set_ylabel("W")
    plt.plot(sample[:,-2], label=meta[-2]["name"], color=meta[-2]["color"])
    plt.fill_between(xrange, 0, sample[:,-2], color=meta[-2]["color"], alpha=0.2)
    plt.plot(sample[:,-1], label=meta[-1]["name"],color=meta[-1]["color"])
    plt.fill_between(xrange, 0, sample[:,-1], color=meta[-1]["color"], alpha=0.4)
    plt.legend(loc='upper right')
    plt.show()

def CsvExport(name,sample,header='Time,T_ext,P_hea,I_sol,T_int'):
    """
    can be used to produce a csv in a timeserie fashion
    with the default header, sample has to be a 4 columns tensor :
    - the 3 sollicitations (outdoor Temp(°C), hvac power(W), sun power(°C),
    - the indoor temp to simulate(°C)

    :sample: data tensor
    :header: colums names separated by comma

    example : CsvExport("test_export",teta,"Time,outdoor temp,kitchen,livingroom,bathroom,bedroom,hvacpower,sunpower")
    """
    datas=np.zeros((teta.shape[0],teta.shape[1]+1))
    datas[:,0]=np.arange(0,sample.shape[0]*step,step)
    for j in range(sample.shape[1]):
        datas[:,j+1]=sample[:,j]
    np.savetxt("{}_{}_step{}s.csv".format(name,house,step),datas,delimiter=',',header=header, comments='')

"""
PREDICTION and OPTIMIZATION SECTION
most of the following methods rely on setting :
    - a truth vector for indoor temperature
    - the number of states to predict : n
for methods calculating the functionnal and its jacobian, you need to set the u tensor which represents the sollicitations :
    - outdoor temp (°C),
    - hvac power(W),
    - solar power(W)
"""
def MatriX(CRES,CS,RI,R0,RF,jac=True):
    A=np.array([ [-1/CRES*(1/RI+1/RF), 1/(CRES*RI)      ],
                 [1/(CS*RI)          , -1/CS*(1/RI+1/R0)] ])

    B=np.array([ [1/(CRES*RF), 1/CRES, 0   ],
                 [1/(CS*R0)  , 0     , 1/CS] ])

    if jac :
        dA=[]
        dA.append(np.array([ [(1/RI+1/RF)/CRES**2, -1/(RI*CRES**2)], [0            ,  0                ] ]))
        dA.append(np.array([ [0                  ,  0             ], [-1/(RI*CS**2), (1/RI+1/R0)/CS**2 ] ]))
        dA.append(np.array([ [1/(CRES*RI**2)     , -1/(CRES*RI**2)], [-1/(CS*RI**2), 1/(CS*RI**2)      ] ]))
        dA.append(np.array([ [0                  , 0              ], [0            , 1/(CS*R0**2)      ] ]))
        dA.append(np.array([ [1/(CRES*RF**2)     , 0              ], [0            , 0                 ] ]))

        dB=[]
        dB.append(np.array([ [-1/(RF*CRES**2), -1/CRES**2, 0], [0            , 0, 0          ] ]))
        dB.append(np.array([ [0              , 0         , 0], [-1/(R0*CS**2), 0, -1/(CS**2) ] ]))
        dB.append(np.array([ [0              , 0         , 0], [0            , 0, 0          ] ]))
        dB.append(np.array([ [0              , 0         , 0], [-1/(CS*R0**2), 0, 0          ] ]))
        dB.append(np.array([ [-1/(CRES*RF**2), 0         , 0], [0            , 0, 0          ] ]))

    if jac :
        return A, B, dA, dB
    else :
        return A, B

def RCpredict_Euler(inputs,CRES,CS,RI,R0,RF,allStates=False):
    """
    inputs is a 3 colums tensor
    column1:T_ext
    column2:P_hea
    column3:I_sol
    """
    A, B = MatriX(CRES,CS,RI,R0,RF,jac=False)
    nbpts=inputs.shape[0]
    H = np.array([[1, 0]])
    # Initialisation of the states
    x = np.zeros((nbpts, n))
    x[0] = np.array((truth[0], truth[0]))
    # Simulation
    for i in range(nbpts-1):
        x[i+1]=np.linalg.inv(np.eye(n)-step*A).dot(x[i]+step*B.dot(inputs[i]))
    # This function returns the first simulated state only
    if allStates == False:
        #return np.dot(H, x.T).flatten()
        return x[:,0]
    else:
        return x

def RCfonc(_p):
    str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(_p)
    print("estimating the fonctionnal - p is {}".format(str))
    CRES=_p[0]
    CS=_p[1]
    RI=_p[2]
    R0=_p[3]
    RF=_p[4]
    x=RCpredict_Euler(u,CRES,CS,RI,R0,RF)
    f=0
    for i in range(len(x)):
        f+=0.5*(x[i]-truth[i])**2
    return f/len(x)

def RCgrad(_p):
    n_par=len(_p)
    str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(_p)
    print("estimating the gradient - p is {}".format(str))
    CRES=_p[0]
    CS=_p[1]
    RI=_p[2]
    R0=_p[3]
    RF=_p[4]
    A, B, dA, dB = MatriX(CRES,CS,RI,R0,RF,jac=True)
    x=RCpredict_Euler(u,CRES,CS,RI,R0,RF,allStates=True)

    z=np.zeros((n,n_par))
    df=np.zeros(n_par)

    for i in range(len(x)-1):
        for j in range(n_par):
            z[:,j]=np.linalg.inv(np.eye(2)-step*A).dot(z[:,j] + step*dA[j].dot(x[i+1]) + step*dB[j].dot(u[i]))
            df[j]+=z[0,j]*(x[i,0]-truth[i])

    return df/len(x)

def RCpredict_Krank(inputs,CRES,CS,RI,R0,RF):
    """
    to solve dx/dt=A(p).x(p,t)+b.u(p,t)
    with x=(Tint,Tenv) as our problem is a 2 states problem
    the enveloppe is unobserved, whereas indoor is nomitored by a temperature sensor
    """
    A, B = MatriX(CRES,CS,RI,R0,RF,jac=False)
    nbpts=inputs.shape[0]
    x=np.zeros((nbpts,n))
    x[0] = np.array((truth[0], truth[0]))
    for i in range(nbpts-1):
        x[i+1]=np.linalg.inv(np.eye(n)-step*A/2).dot((np.eye(n)+step*A/2).dot(x[i])+step*B.dot(inputs[i+1]+inputs[i])/2)
    H = np.array([[1, 0]])
    return np.dot(H, x.T).flatten()

def RCAdjointState(_p):
    CRES=_p[0]
    CS=_p[1]
    RI=_p[2]
    R0=_p[3]
    RF=_p[4]
    A, B = MatriX(CRES,CS,RI,R0,RF,jac=False)
    x=RCpredict_Krank(u,CRES,CS,RI,R0,RF)
    S=np.zeros((len(x),n))
    for j in range(len(x)-2,-1,-1):
        S[j]=np.linalg.inv(np.eye(n)/step-A.T/2).dot((np.eye(n)/step+A.T/2).dot(S[j+1]) + np.array([x[j]-truth[j],0])/len(x))
    return S


nbptinh=12
step=3600//nbptinh
house="temoin"
params=[ {"id":1,"name":"outdoor temp","color":"blue","action":"smp"},
         {"id":182,"name": "kitchen","color":"purple","action":"smp"},
         {"id":191,"name":"livingroom","color":"orange","action":"smp"},
         {"id":185,"name":"bathroom","color":"green","action":"smp"},
         {"id":188,"name":"bedroom","color":"#b6e91f","action":"smp"},
         {"id":139,"name":"hvac power (W)","color":"red","action":"smp"}]
smpStart=1556040600
tDays=14
# PHPFina index for the indoor temperature truth
truth_id=2

teta=GoToTensor(params,step,smpStart,tDays*24*nbptinh)

# generate some sun and amend the params list
#calculating the starting hour for the datarange
smpH=datetime.utcfromtimestamp(smpStart).hour
teta[:,-1]=generateSunRange(1000,nbptinh, teta.shape[0], smpH)
params.append({"name":"solar power (W)","color":"yellow"})

#if yu dont want to visualize the full dataset, yu can select only some columns
meta=[params[0],params[truth_id],params[5],params[6]]
dataset = np.vstack([teta[:,0],teta[:,truth_id],teta[:,5],teta[:,6]]).T
print(dataset.shape)

p0=[3.67e+5,8.95e+7,1e-2,1e-2,1e-2]
#p0=[4e+5,9e+7,1e-2,1e-2,1e-2]

# u and truth will be used by RCpredict, RCfonc and RCgrad
# u is the inputs/sollicitations tensor with 3 columns : T_ext ,P_hea, I_sol
u = np.vstack([teta[:,0],teta[:,5],teta[:,6]]).T
# we set the truth
truth=teta[:,truth_id]
# we have 2 states : indoor and envelope - envelope is unobserved
# n is the number of states
n = 2

Tint_simEuler=RCpredict_Euler(u,p0[0],p0[1],p0[2],p0[3],p0[4])
Tint_simKrank=RCpredict_Krank(u,p0[0],p0[1],p0[2],p0[3],p0[4])
S=RCAdjointState(p0)
visualize(dataset,meta,house,Tint_simEuler=Tint_simEuler,Tint_simKrank=Tint_simKrank,S0=S[:,0])

from scipy import optimize
bounds=[(0,np.inf),(0,np.inf),(1e-5,1e-1),(1e-5,1e-1),(1e-5,1e-1)]

res=optimize.minimize(RCfonc, p0, method="BFGS", jac=RCgrad, bounds=bounds)
print(res)
popt=res["x"]

input("press key to vizualize the optimized indoor temperature curve")
Tint_optBFGS=RCpredict_Euler(u, popt[0], popt[1], popt[2], popt[3], popt[4])
visualize(teta,params,house,Tint_optBFGS=Tint_optBFGS)
