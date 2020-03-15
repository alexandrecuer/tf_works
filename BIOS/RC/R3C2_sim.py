import numpy as np
from PHPFina import PHPFina,humanDate
import matplotlib.pylab as plt
from datetime import datetime
import random
import math
import copy
from math import floor, log10

debug=False

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

def visualize(sample,lib,**kwargs):
    lab=["out. temp","indoor temp", "Hvac W", "Sun W"]
    col=["blue","green","red","orange"]
    xrange=np.arange(sample.shape[0])
    ax1=plt.subplot(111)
    ax1.set_ylabel("°C")
    plt.title(lib)
    plt.plot(sample[:,1],label=lab[1], color=col[1])
    plt.plot(sample[:,0],label=lab[0], color=col[0])
    if len(kwargs):
        # you can plot 4 extra curves
        icons=[':','--','o','*']
        indice=0
        for key, vals in kwargs.items():
            plt.plot(vals,icons[indice],markersize=2, label="{}.".format(key))
            indice+=1
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.set_ylabel("W")
    plt.plot(sample[:,2], label=lab[2], color=col[2])
    plt.fill_between(xrange, 0, sample[:,2], color=col[2], alpha=0.2)
    plt.plot(sample[:,3], label=lab[3],color=col[3])
    plt.fill_between(xrange, 0, sample[:,3], color=col[3], alpha=0.4)
    plt.legend(loc='upper right')
    plt.show()

def calcG(sample,p):
    """
    value of the G function given a vector p of parameters (to optimize)

    :sample: truths vector on time i [Text, Tint, PHVAC, Psun] OR a truths array for a full time range
    :p: parameters vector [cres,cs,ri,r0,rf]
    :return: a numpy vector of size 2 with the G values for the given sample OR a numpy array of size (sample.shape[0],2)

    please note Tint is given but not involved in the calculation process
    """
    if len(sample.shape)>1:
        g=np.zeros((2, sample.shape[0]))
        g[0,:]=sample[:,2]/p[0]+sample[:,0]/(p[0]*p[4])
        g[1,:]=sample[:,3]/p[1]+sample[:,0]/(p[1]*p[3])
    else:
        g=np.zeros(2)
        g[0]=sample[2]/p[0]+sample[0]/(p[0]*p[4])
        g[1]=sample[3]/p[1]+sample[0]/(p[1]*p[3])
    return g

def calcdG(sample,p):
    """
    derivation of the G function with respect to p

    :sample: truths vector on time i [Text, Tint, PHVAC, Psun]
    :p: parameters vector [cres,cs,ri,r0,rf]
    :return: a numpy array with the derivatives values for the given sample
    dg[index] is the derivative with respect to p[index]
    """
    dg=np.zeros((len(p),2))
    # we now calculate all the dg elements that are different from zeros
    dg[0,0]=-(sample[2]+sample[0]/p[4])/p[0]**2
    dg[1,1]=-(sample[3]+sample[0]/p[3])/p[1]**2
    dg[3,1]=-sample[0]/(p[1]*p[3]**2)
    dg[4,0]=-sample[0]/(p[0]*p[4]**2)
    return dg

def calcA(p):
    """
    the A matrix given a p vector
    """
    A=np.zeros((2,2))
    A[0,0]=-(1/p[2]+1/p[4])/p[0]
    A[0,1]=1/(p[2]*p[0])
    A[1,0]=1/(p[2]*p[1])
    A[1,1]=-(1/p[2]+1/p[3])/p[1]
    return A

def calcB(p):
    """
    the B=I-step*A matrix given a p vector
    """
    I=np.eye(2)
    B=I-step*calcA(p)
    return B

def calcdA(p):
    dA=np.zeros((len(p),2,2))
    # we now calculate all the dA elements that are different from zero
    #dA/dcres
    dA[0,0,0]=(1/p[2]+1/p[4])/p[0]**2
    dA[0,0,1]=-1/(p[2]*p[0]**2)
    #dA/dcs
    dA[1,1,0]=-1/(p[2]*p[1]**2)
    dA[1,1,1]=(1/p[2]+1/p[3])/p[1]**2
    #dA/dri
    dA[2,0,0]=1/(p[0]*p[2]**2)
    dA[2,0,1]=-1/(p[0]*p[2]**2)
    dA[2,1,0]=-1/(p[1]*p[2]**2)
    dA[2,1,1]=1/(p[1]*p[2]**2)
    #dA/dr0
    dA[3,1,1]=1/(p[1]*p[3]**2)
    #dA/drf
    dA[4,0,0]=1/(p[0]*p[4]**2)
    return dA

def RC_model_deterministic(inputs,CRES,CS,RI,R0,RF):
    """
    inputs is a 3 colums tensor
    column1:T_ext
    column2:P_hea
    column3:I_sol
    """
    A=np.array([[-1/CRES*(1/RI+1/RF), 1/(CRES*RI)],
                   [1/(CS*RI), -1/CS*(1/RI+1/R0)]])
    B=np.array([[1/(CRES*RF), 1/CRES, 0],
                   [1/(CS*R0), 0, 1/CS]])
    H = np.array([[1, 0]])
    # we have 2 states : indoor and envelope - envelope is unobserved
    # n is the number of states
    n = 2
    # Initialisation of the states
    x = np.zeros((len(inputs), n))
    x[0] = np.array((teta[0,1], teta[0,1]))
    # Simulation
    for i in range(len(inputs)-1):
        x[i+1]=np.linalg.inv(np.eye(n)-step*A).dot(x[i]+step*B.dot(inputs[i]))
    # This function returns the first simulated state only
    return np.dot(H, x.T).flatten()

def RCgradient(u,CRES,CS,RI,R0,RF):
    print("I am supposed to be the gradient")
    A=np.array([ [-1/CRES*(1/RI+1/RF), 1/(CRES*RI)      ],
                 [1/(CS*RI)          , -1/CS*(1/RI+1/R0)] ])
    dA=[]
    dA.append(np.array([ [(1/RI+1/RF)/CRES**2, -1/(RI*CRES**2)], [0            ,  0                ] ]))
    dA.append(np.array([ [0                  ,  0             ], [-1/(RI*CS**2), (1/RI+1/R0)/CS**2 ] ]))
    dA.append(np.array([ [1/(CRES*RI**2)     , -1/(CRES*RI**2)], [-1/(CS*RI**2), 1/(CS*RI**2)      ] ]))
    dA.append(np.array([ [0                  , 0              ], [0            , 1/(CS*R0**2)      ] ]))
    dA.append(np.array([ [1/(CRES*RF**2)     , 0              ], [0            , 0                 ] ]))

    B=np.array([[1/(CRES*RF), 1/CRES, 0],
                   [1/(CS*R0), 0, 1/CS]])
    dB=[]

    print(dA)

def Krank_Nicholson(u,CRES,CS,RI,R0,RF):
    """
    to solve dx/dt=A(p).x(p,t)+b.u(p,t)
    with x=(Tint,Tenv) as our problem is a 2 states problem
    the enveloppe is unobserved, whereas indoor is nomitored by a temperature sensor
    """
    A=np.array([[-1/CRES*(1/RI+1/RF), 1/(CRES*RI)],
                   [1/(CS*RI), -1/CS*(1/RI+1/R0)]])
    B=np.array([[1/(CRES*RF), 1/CRES, 0],
                   [1/(CS*R0), 0, 1/CS]])
    nbpts=u.shape[0]
    n=2
    x=np.zeros((nbpts,n))
    x[0] = np.array((teta[0,1], teta[0,1]))
    for i in range(nbpts-1):
        x[i+1]=np.linalg.inv(np.eye(n)-step*A/2).dot((np.eye(n)+step*A/2).dot(x[i])+step*B.dot(u[i+1]+u[i])/2)
    H = np.array([[1, 0]])
    return np.dot(H, x.T).flatten()

def eulerDis(sample, p, gradient=True,*args):
    # args[0] is the provided synthetic truth for validation purposes
    # if not given, we take the truth from the teta tensor (column 1)
    if args:
        truth=args[0]
    else:
        truth=sample[:,1]
    nbpts=sample.shape[0]
    T=np.zeros((nbpts,2))
    T[0,:]=np.array([truth[0],truth[0]])
    f=0
    C=np.linalg.inv(calcB(p))
    if gradient:
        df=np.zeros(len(p))
        z=np.zeros((2,len(p)))
        dA=calcdA(p)

    for i in range(nbpts-1):
        T[i+1,:]=C.dot(T[i,:]+step*calcG(sample[i,:],p))
        f+=0.5*(T[i+1,0]-truth[i+1])**2
        if gradient:
            dG=calcdG(sample[i,:],p)
            reguld=np.array(6)
            for j in range(len(p)):
                z[:,j]=C.dot(z[:,j]+step*(dA[j].dot(T[i+1,:])+dG[j]))
            df+=z[0,:]*(T[i+1,0]-truth[i+1])

    if gradient:
        return T, f/nbpts, df/nbpts
    else:
        return T, f/nbpts

def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def regul(f):
    ref=np.zeros(len(f))
    for i in range(len(f)):
        ref[i]=10**(-fexp(f[i]))
    return ref

def gradient_descent(sample,p):
    a=0.5
    f=[]
    quality=[]
    nbgradit=100
    tab_p=np.zeros((nbgradit,len(p)))
    it=0
    while it < nbgradit:
        print("starting gradient iteration {} with a={} and p={}".format(it,a,p))
        tab_p[it]=p
        Tit, fit, dfit = eulerDis(sample,p,gradient=True)
        f.append(fit)
        if fit<1E-5:
            break
        print("after gradient calculation {}, f is {}".format(it,fit))
        print("gradient is {}".format(dfit))
        if it==0:
            a=a*regul(dfit)
            print(a)
        input("press key")
        if it > 0:
            q=fit/f[0]
            print("q is {}".format(q))
            quality.append(q)
            if q<1E-6:
                print("we are checking out with f/f0={}".format(q))
                break
        cuts=0
        fb=fit
        while fb >= fit :
            cuts+=1
            pb=p-a*dfit
            print("calculating f with p={} for a={}".format(pb,a))
            Tb, fb = eulerDis(sample,pb,gradient=False)
            print("f is {}".format(fb))
            #input("press key")
            if fb <= fit :
                a = 1.1*a
                break
            else :
                a = 0.5*a
        print("end of iteration {} with a={} after {} half-cuts".format(it,a,cuts))
        #input("press any key")
        p=p-a*dfit
        it+=1
    return tab_p, f, it-1

nbptinh=2
step=3600//nbptinh
# house="temoin"
#params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"smp"}]
house="ite"
params=[{"id":1,"action":"smp"},{"id":167,"action":"smp"},{"id":145,"action":"smp"}]

# Monday 17 December 2018 00:00:00
#smpStart=1545004800
#tDays=40
smpStart=1547121600
tDays=15

smpH=datetime.utcfromtimestamp(smpStart).hour
print(smpH)
teta=GoToTensor(params,step,smpStart,tDays*24*nbptinh)

# in W
qs=1000
teta[:,-1]=generateSunRange(qs,nbptinh, teta.shape[0], smpH)
visualize(teta,house)
"""
# this is the exportation process to produce csv
datas=np.zeros((teta.shape[0],teta.shape[1]+1))
datas[:,1]=teta[:,0]
datas[:,2]=teta[:,2]
datas[:,3]=teta[:,3]
datas[:,4]=teta[:,1]
datas[:,0]=np.arange(0,teta.shape[0]*step,step)
np.savetxt("datas_{}_step{}s.csv".format(house,step),datas,delimiter=',',header='Time,T_ext,P_hea,I_sol,T_int', comments='')
"""


cres=4e+5
cs=9e+7
ri=1e-2
r0=1e-2
rf=1e-2
p=[cres,cs,ri,r0,rf]

# u is the inputs tensor with 3 columns : T_ext ,P_hea, I_sol
u = np.vstack([teta[:,0],teta[:,2],teta[:,3]]).T
T_in_det=RC_model_deterministic(u,cres,cs,ri,r0,rf)
Temp, functionf = eulerDis(teta, p, gradient=False)
Tk=Krank_Nicholson(u,cres,cs,ri,r0,rf)

visualize(teta,house,Tint_Euler=T_in_det,Tint_krank_nicholson=Tk)
RCgradient(u,cres,cs,ri,r0,rf)

input("press key")

from scipy.optimize import curve_fit

#bounds=(0,np.inf)
bounds=([0,0,0,0,0],[np.inf,np.inf,10,10,10])

"""
see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
"""
init_guess = [4e5, 9e7, 1e-2, 1e-2, 1e-2]
popt, pcov = curve_fit(Krank_Nicholson,
                       xdata = u,
                       ydata = teta[:,1],
                       p0 = init_guess,
                       method='trf',
                       bounds=bounds)
# Standard deviation of the parameter estimates
stdev = np.diag(pcov)**0.5
print(popt)
print("standart deviation is {}".format(stdev))
Tin_trf_opt=Krank_Nicholson(u, popt[0], popt[1], popt[2], popt[3], popt[4])

visualize(teta,house,Tin_trf_opt=Tin_trf_opt)

#Tuesday 2 October 2018 00:00:00
vStart=1538438400
smpH=datetime.utcfromtimestamp(vStart).hour
print(smpH)
validation=GoToTensor(params,step,vStart,110*24*nbptinh)
# adding some sun
validation[:,-1]=generateSunRange(qs,nbptinh, validation.shape[0], smpH)
uval = np.vstack([validation[:,0],validation[:,2],validation[:,3]]).T
Tin_sim=Krank_Nicholson(uval, popt[0], popt[1], popt[2], popt[3], popt[4])

visualize(validation,house,Tin_sim=Tin_sim)

"""
synthetic=copy.deepcopy(teta)
synthetic[:,1]=Temp[:,0]
visualize(synthetic,house)

#p0=[2*cres,1.1*cs,ri,4*r0,2*rf]
p0=[6*cres,cs,ri,r0*1.1,rf*1.2]

tab, fonc, nbits = gradient_descent(synthetic,p0)
solution=tab[nbits]
print("solution is {}".format(solution))
print("we have an f value of {} when we started with {}".format(fonc[nbits],fonc[0]))

Tsim, fsim = eulerDis(synthetic, tab[nbits], gradient=False)
visualize(synthetic,house,Tsim[:,0])
"""
