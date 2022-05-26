import numpy as np
from PHPFina import PHPFina,humanDate
import matplotlib.pylab as plt


def InitializeFeed(nb,step,start):
    feed=PHPFina(nb,step)
    feed.getMetas()
    feed.setStart(start)
    return feed

# given some PHPFina feeds, a period and a start (as a unix timestamp) both in seconds
def GoToTensor(params,step,start,nbsteps):
    #print("going to tensor for {} feeds".format(len(params)))
    float_data=np.zeros((nbsteps,len(params)))
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

nbptinh=12
step=3600//nbptinh
params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"smp"}]
temoin=GoToTensor(params,step,1537876800,148*24*nbptinh)

ax1=plt.subplot(111)
ax1.set_ylabel("°C")
plt.title("pavillon témoin")
plt.plot(temoin[:,1],label="tint")
plt.plot(temoin[:,0],label="text")
ax1.tick_params(axis='y')
plt.legend()
ax2 = ax1.twinx()
ax2.set_ylabel("W")
plt.plot(temoin[:,2], color="green", label="power")
plt.legend()
plt.show()

# building volume in m3
vb=300
# air bulk density in kg/m3
rho_air=1.22
# air heat capacity in J/(kg.K)
c_air=1004
# heated floor area in m2
floor=80
# inertia in J/(K.m2)
inertia = 200000
# q4pasurf in m3/(h.m2)
q4pasurf=1.2
# atbat in m2 - off-floor loss area
atbat=217

# cs in J/K
cs=inertia*floor*10
# res in J/K
cres=c_air*rho_air*vb*10
# leakage resistance in K/W
rf=3600/(rho_air*c_air*q4pasurf*atbat*5)
# internal wall resistance in K/W
ri=1/(3*(atbat+floor))
# external wall resistance in K/W
r0=1/(2*(atbat+floor))

# in W
qs=6500

def alpha(r,c,*args):
    return (-1/r-1/args[0])/c

def beta(r,*args):
    return 1/(r*args[0])

def calcG(sample,x_dev):
    g=np.zeros(2)
    g[0]=sample[2]/x_dev[0]+sample[0]/(x_dev[0]*x_dev[4])
    g[1]=x_dev[5]/x_dev[1]+sample[0]/(x_dev[1]*x_dev[3])
    return g

# x_dev is [cres,cs,ri,r0,rf,qs]
def calcdG(sample,x_dev):
    dg=np.zeros((6,2))
    dg[0,0]=-(sample[2]+sample[0]/x_dev[4])/x_dev[0]**2
    dg[0,1]=0
    dg[1,0]=0
    dg[1,1]=-(x_dev[5]+sample[0]/x_dev[3])/x_dev[1]**2
    dg[2,0]=dg[2,1]=0
    dg[3,0]=0
    dg[3,1]=-sample[0]/(x_dev[1]*x_dev[3]**2)
    dg[4,0]=-sample[0]/(x_dev[0]*x_dev[4]**2)
    dg[4,1]=0
    dg[5,0]=0
    dg[5,1]=1/x_dev[1]
    return dg

def calcB(x_dev):
    I=np.eye(2)
    A=np.zeros((2,2))
    A[0,0]=alpha(x_dev[2],x_dev[0],x_dev[4])
    A[0,1]=beta(x_dev[2],x_dev[0])
    A[1,0]=beta(x_dev[2],x_dev[1])
    A[1,1]=alpha(x_dev[2],x_dev[1],x_dev[3])
    B=I-step*A
    return B

def calcdA(x_dev):
    dA=np.zeros((6,2,2))
    #dA/dcres
    dA[0,0,0]=alpha(x_dev[2],-x_dev[0]**2,x_dev[4])
    dA[0,0,1]=beta(x_dev[2],-x_dev[0]**2)
    dA[0,1,0]=dA[0,1,1]=0
    #dA/dcs
    dA[1,0,0]=dA[1,0,1]=0
    dA[1,1,0]=beta(x_dev[2],-x_dev[1]**2)
    dA[1,1,1]=alpha(x_dev[2],-x_dev[1]**2,x_dev[3])
    #dA/dri
    dA[2,0,0]=beta(x_dev[2]**2,x_dev[0])
    dA[2,0,1]=-dA[2,0,0]
    dA[2,1,0]=beta(-x_dev[2]**2,x_dev[1])
    dA[2,1,1]=-dA[2,1,0]
    #dA/dr0
    dA[3,0,0]=dA[3,0,1]=dA[3,1,0]=0
    dA[3,1,1]=beta(-x_dev[3]**2,x_dev[1])
    #dA/drf
    dA[4,0,0]=beta(-x_dev[4]**2,x_dev[0])
    dA[4,0,1]=dA[4,1,0]=dA[4,1,1]=0
    #dA/dqs
    dA[5,0,0]=dA[5,0,1]=dA[5,1,0]=dA[5,1,1]=0
    return dA

def eulerDis(nbpts,x_dev):
    T=np.zeros((nbpts,2))
    T[0,:]=np.array([temoin[0,1],temoin[0,1]])
    f=0
    for i in range(nbpts-1):
        T[i+1,:]=np.linalg.inv(calcB(x_dev)).dot(T[i,:]+step*calcG(temoin[i,:],x_dev))
        f+=(T[i+1,0]-temoin[i+1,1])**2
    return T, f

def gradient_descent(x_dev):
    a=0.7
    f=[]
    df=[]
    for it in range(1000):
        print("starting gradient iteration {} with a={}".format(it,a))
        f_it=0
        df_it=np.zeros(6)
        T_it=np.zeros((nbpts,2))
        T_it[0,:]=np.array([temoin[0,1],temoin[0,1]])


nbpts=40000
x0=[cres,cs,ri,r0,rf,qs]

Temp, err = eulerDis(nbpts,x0)

plt.subplot(111)
plt.title("pavillon témoin")
plt.plot(temoin[:,1],label="tint true", color="red")
plt.plot(temoin[:,0],label="text true", color="blue")
plt.plot(Temp[:,0],label="Tint", color="orange")
plt.plot(Temp[:,1],label="Tsurface", color="green")
plt.legend()
plt.show()
