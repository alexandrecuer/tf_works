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
            sun[i]=max(0,qs_max*(math.exp(-10*((i-sunrise-zenith)/sunstp)**2)))
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
        icons=[':','--','--','*']
        indice=0
        for key, vals in kwargs.items():
            plt.plot(vals,icons[indice],markersize=2, label="{}.".format(key))
            indice+=1
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.set_ylabel("W")
    plt.fill_between(xrange, 0, sample[:,-2], label=meta[-2]["name"], color=meta[-2]["color"], alpha=0.2)
    plt.fill_between(xrange, 0, sample[:,-1], label=meta[-1]["name"], color=meta[-1]["color"], alpha=0.4)
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
    datas=np.zeros((sample.shape[0],sample.shape[1]+1))
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
    # we could generate N+1 points with our N sollicitations
    # but for the last point we will not be able to evaluate the functionnal
    x = np.zeros((nbpts, n))
    x[0] = np.array([truth[0], u[0,0]+offset])
    # Simulation
    for i in range(nbpts-1):
        x[i+1]=np.linalg.inv(np.eye(n)-step*A).dot(x[i]+step*B.dot(inputs[i]))

    if allStates == False:
        return x[:,0]
    else:
        return x

def RCpredict_Krank(inputs,CRES,CS,RI,R0,RF,allStates=False):
    """
    to solve dx/dt=A(p).x(p,t)+b.u(p,t)
    with x=(Tint,Tenv) as our problem is a 2 states problem
    the enveloppe is unobserved, whereas indoor is nomitored by a temperature sensor
    """
    A, B = MatriX(CRES,CS,RI,R0,RF,jac=False)
    nbpts=inputs.shape[0]
    x=np.zeros((nbpts,n))
    x[0] = np.array([truth[0], u[0,0]+offset])

    AS_B=np.linalg.inv(np.eye(n)-step*A/2)
    AS_C=AS_B.dot(np.eye(n)+step*A/2)
    AS_B=step*AS_B/2

    for i in range(nbpts-1):
        x[i+1]=AS_C.dot(x[i])+AS_B.dot(B.dot(inputs[i+1]+inputs[i]))

    if allStates == False:
        #H = np.array([[1, 0]])
        #return np.dot(H, x.T).flatten()
        return x[:,0]
    else:
        return x

def RCfonc(_p,type="classic"):
    str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(_p)
    print("estimating the fonctionnal - p is {}".format(str))
    CRES=_p[0]
    CS=_p[1]
    RI=_p[2]
    R0=_p[3]
    RF=_p[4]
    if type=="classic":
        x=RCpredict_Euler(u,CRES,CS,RI,R0,RF)
    elif type=="krank":
        x=RCpredict_Krank(u,CRES,CS,RI,R0,RF)
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

def RCgrad_Krank(_p):
    n_par=len(_p)
    str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(_p)
    print("estimating the gradient - p is {}".format(str))
    CRES=_p[0]
    CS=_p[1]
    RI=_p[2]
    R0=_p[3]
    RF=_p[4]
    A, B, dA, dB = MatriX(CRES,CS,RI,R0,RF,jac=True)
    x=RCpredict_Krank(u,CRES,CS,RI,R0,RF,allStates=True)

    AS_B=np.linalg.inv(np.eye(n)-step*A/2)
    AS_C=AS_B.dot(np.eye(n)+step*A/2)
    AS_B=step*AS_B/2

    S=np.zeros((len(x),n))
    for j in range(len(x)-1,1,-1):
        S[j-1]=S[j].T.dot(AS_C)+np.array([x[j,0]-truth[j],0]).T.dot(AS_B)*2/len(x)

    df=np.zeros(n_par)
    for i in range(len(x)-1):
        for j in range(n_par):
            df[j]+=S[i].dot(dA[j].dot(x[i+1]+x[i])+dB[j].dot(u[i+1,:]+u[i,:]))/2
    return df

# number of points in an hour
nbptinh=2
step=3600//nbptinh
# offset in °C to add to the initial guess for the envelope temperature when starting prediction
offset=0

house="ite"
params=[ {"id":1,"name":"outdoor temp","color":"blue","action":"smp"},
         {"id":170,"name": "kitchen","color":"purple","action":"smp"},
         {"id":167,"name": "livingroom","color":"orange","action":"smp"},
         {"id":173,"name":"bathroom","color":"green","action":"smp"},
         {"id":176,"name":"bedroom","color":"#b6e91f","action":"smp"},
         {"id":145,"name":"hvac power (W)","color":"red","action":"smp"}]
smpStart=1547121600
#smpStart=1545902400
tDays=15
#tDays=30
"""

house="temoin"
params=[ {"id":1,"name":"outdoor temp","color":"blue","action":"smp"},
         {"id":182,"name": "kitchen","color":"purple","action":"smp"},
         {"id":191,"name":"livingroom","color":"orange","action":"smp"},
         {"id":185,"name":"bathroom","color":"green","action":"smp"},
         {"id":188,"name":"bedroom","color":"#b6e91f","action":"smp"},
         {"id":139,"name":"hvac power (W)","color":"red","action":"smp"}]
smpStart=1556040600
tDays=14
"""
# we set the truth choosing an id among the indoor temperature feeds we have collected (1,2,3,4)
truth_id=2
# algo can be krank with an evaluation of the gradient on the basis of the adjoint state or a classic scheme
#algo="classic"
algo="classic"

#scaling
p0=np.array([1e+6,1e+7,1e-2,1e-2,1e-2])
#initial guess
x0=np.array([9,9,1,1,1])

# fetching the feeds
teta=GoToTensor(params,step,smpStart,tDays*24*nbptinh)
# generate some sun and amend the params list
# calculating the starting hour for the datarange
smpH=datetime.utcfromtimestamp(smpStart).hour
teta[:,-1]=generateSunRange(500,nbptinh, teta.shape[0], smpH)
params.append({"name":"solar power (W)","color":"yellow"})

# u and truth will be used by RCpredict, RCfonc and RCgrad
# u is the inputs/sollicitations tensor with 3 columns : T_ext ,P_hea, I_sol
u = np.vstack([teta[:,0],teta[:,5],teta[:,6]]).T
truth=teta[:,truth_id]
# n is the number of states
# we have 2 states : indoor and envelope - envelope is unobserved
n = 2
# proceed a selection to avoid visualization of the full dataset
meta=[params[0],params[truth_id],params[-2],params[-1]]
dataset = np.vstack([teta[:,0],teta[:,truth_id],teta[:,-2],teta[:,-1]]).T


s=x0*p0
T_simEuler=RCpredict_Euler(u,s[0],s[1],s[2],s[3],s[4],allStates=True)
Tint_simKrank=RCpredict_Krank(u,s[0],s[1],s[2],s[3],s[4])
visualize(dataset,meta,house,Tint_simEuler=T_simEuler[:,0],TS_simEuler=T_simEuler[:,1],Tint_simKrank=Tint_simKrank)


from scipy import optimize
#for method L-BFGS-B
bounds=[(0,np.inf),(0,np.inf),(0,1),(0,1),(0,1)]

# we will use array x to store the evolution of the parameters during the iteration process
# they will stand as quality indicators for convergence or not
x=[]

# initiate functions with regularisation
def fonc(_x):
    return RCfonc(p0*_x,type=algo)

def grad(_x):
    x.append(_x)
    if algo=="krank":
        return p0*RCgrad_Krank(p0*_x)
    elif algo=="classic":
        return p0*RCgrad(p0*_x)

#res=optimize.minimize(fonc, x0, method="L-BFGS-B", jac=grad, bounds=bounds)
res=optimize.minimize(fonc, x0, method="BFGS", jac=grad)

# SANITY CONVERGENCE CHECK
quality=np.array(x)
nb=321
lib=["cres", "cs", "ri", "r0", "rf"]
for z in range(len(lib)):
    str="%.0E" % (1/p0[z])
    lib[z]="{} x {}".format(lib[z],str)
#it is the iteration number
for it in range(quality.shape[-1]):
    plt.subplot(nb)
    nb+=1
    plt.plot(quality[:,it],label=lib[it])
    plt.legend()
plt.show()

#res=optimize.minimize(RCfonc, p0, method="BFGS", jac=RCgrad, bounds=bounds)
print(res)
popt=res["x"]*p0
print(popt)

input("press key to vizualize the optimized indoor temperature curve")
if algo=="krank":
    T_opt=RCpredict_Krank(u, popt[0], popt[1], popt[2], popt[3], popt[4],allStates=True)
elif algo=="classic":
    T_opt=RCpredict_Euler(u, popt[0], popt[1], popt[2], popt[3], popt[4],allStates=True)
visualize(teta,params,house,Tint_opt=T_opt[:,0],Ts_opt=T_opt[:,1])

#lets make a simulation out of the optimisation window
smpStart=1540166400
tDays=4
# we do not give the sun to GoToTensor as it was not monitored by Themis
winter=GoToTensor(params[:-1],step,smpStart,tDays*24*nbptinh)
smpH=datetime.utcfromtimestamp(smpStart).hour
#adding the sun
winter[:,-1]=generateSunRange(500,nbptinh, winter.shape[0], smpH)
#setting the sollicitations and the truth
u = np.vstack([winter[:,0],winter[:,5],winter[:,6]]).T
truth=winter[:,truth_id]
if algo=="krank":
    T_sim_winter=RCpredict_Krank(u, popt[0], popt[1], popt[2], popt[3], popt[4],allStates=True)
elif algo=="classic":
    T_sim_winter=RCpredict_Euler(u, popt[0], popt[1], popt[2], popt[3], popt[4],allStates=True)
meta=[params[0],params[truth_id],params[-2],params[-1]]
dataset = np.vstack([winter[:,0],winter[:,truth_id],winter[:,-2],winter[:,-1]]).T
visualize(dataset,meta,house,Tint_sim=T_sim_winter[:,0],Ts_sim=T_sim_winter[:,1])
