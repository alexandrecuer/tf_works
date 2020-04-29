"""
RC modelization toolkit
"""
import numpy as np
from src.tools import PHPFina
#from .phpfina import PHPFina
import matplotlib.pylab as plt
import matplotlib.animation as animation
from datetime import datetime
import random
import math
import copy
from scipy import optimize

def generateSunDay(qs_max, nb):
    """
    **TO USE ONLY IF YOU CANNOT GENERATE A SYNTHETIC SUN FEED**

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
    **TO USE ONLY IF YOU CANNOT GENERATE A SYNTHETIC SUN FEED**

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

dir="phpfina"

def InitializeFeed(nb,step,start,dir=dir):
    feed=PHPFina(nb,step,dir)
    feed.getMetas()
    feed.setStart(start)
    return feed

def GoToTensor(params,step,start,nbsteps,dir=dir, FINASun=False):
    """
    CUSTOM GoToTensor method

    given some PHPFina feeds, a period and a start (as a unix timestamp) both in seconds

    if FINASun is True, we have the sun among the PHPFina feeds

    if FINASun is False, we will have to add it later, so we extend the matrix with one more column
    """
    if FINASun:
        float_data=np.zeros((nbsteps,len(params)))
    else:
        float_data=np.zeros((nbsteps,len(params)+1))
    for i in range(len(params)):
        #print("feed number {}".format(i))
        feed=InitializeFeed(params[i]["id"],step,start,dir)
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

     - the 2 last colums are hvac power and sun power in W

     - all other columns are temperature fields

    :meta: metadatas list - the minimal infos are name and color :

    ```
    [{"name":"first curve","color":"green"},....]
    ```

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

def CsvExport(name,step,sample,header='Time,T_ext,P_hea,I_sol,T_int'):
    """
    can be used to produce a csv in a timeserie fashion

    with the default header, sample has to be a 4 columns tensor :

    - the 3 sollicitations (outdoor Temp(°C), hvac power(W), sun power(°C),

    - the indoor temp to simulate(°C)

    :step: interval in seconds

    :sample: data tensor

    :header: colums names separated by comma

    example : CsvExport("test_export",1800,teta,"Time,outdoor temp,kitchen,livingroom,bathroom,bedroom,hvacpower,sunpower")
    """
    datas=np.zeros((sample.shape[0],sample.shape[1]+1))
    datas[:,0]=np.arange(0,sample.shape[0]*step,step)
    for j in range(sample.shape[1]):
        datas[:,j+1]=sample[:,j]
    np.savetxt("{}_{}_step{}s.csv".format(name,house,step),datas,delimiter=',',header=header, comments='')


def MatriX(p,jac=True):
    """
    The RC matrix associated to a R3C2 electric model of a Building

    CRES: thermal capacity of the indoor (the air inside the building)

    CS: thermal capacity of the envelope

    RI: thermal resistance between the envelope and the indoor (wall internal resistance)

    R0: thermal resistance between the envelope and the outdoor (wall external resistance)

    RF: thermal resistance due to air leakage
    """
    #print(p)
    CRES=p[0]
    CS=p[1]
    RI=p[2]
    R0=p[3]
    RF=p[4]
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

def RCpredict_Euler(step, p, x0, inputs, allStates=False):
    """
    make predictions with the Euler explicit scheme

    ### DIFFERENTIAL SYSTEM TOOLKIT

    dx/dt=A(p).x(p,t)+B.inputs(p,t)

    p is the parameter vector we want to optimize

    x=(T_int,T_env) as our problem is a 2 states problem

    The enveloppe is unobserved, whereas indoor is monitored by a temperature sensor

    *if allStates is False (default) only return the observed state !*

    x0 is the initial guess

    inputs is a 3 colums tensor which represents the sollicitations :

     - column1:T_ext / outdoor temp (°C),

     - column2:P_hea / hvac power(W),

     - column3:I_sol / solar power(W)

    T_ext and P_hea are monitored, I_sol is not easy to acquire, so we will use a simulation

    2 discretization functions :

     - RCpredict_Euler(step,p,x0,inputs,allStates=False)

     - RCpredict_Krank(step,p,x0,inputs,allStates=False), which uses the  Krank Nicholson scheme

    all other methods rely on setting a truth variable in addition to p,x0 and inputs

     - RCfonc(step,p, x0, inputs, truth, type="classic", verbose=True) is the cost function

     - RCgrad(step,p, x0, inputs, truth, verbose=True)

     - RCgrad_Krank(step,p, x0, inputs, truth, verbose=True)

    truth represents field reality for indoor temperature

    """

    A, B = MatriX(p,jac=False)
    nbpts=inputs.shape[0]
    n=x0.shape[0]
    # Initialisation of the states
    # we could generate N+1 points with our N sollicitations
    # but for the last point we will not be able to evaluate the functionnal
    x = np.zeros((nbpts, n))
    x[0] = x0
    # Simulation
    for i in range(nbpts-1):
        x[i+1]=np.linalg.inv(np.eye(n)-step*A).dot(x[i]+step*B.dot(inputs[i]))

    if allStates == False:
        return x[:,0]
    else:
        return x

def RCpredict_Krank(step, p, x0, inputs, allStates=False):
    """
    make predictions with the krank nichoson scheme
    """
    A, B = MatriX(p,jac=False)
    nbpts=inputs.shape[0]
    n=x0.shape[0]
    x=np.zeros((nbpts,n))
    x[0] = x0

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

def RCfonc(step, p, x0, inputs, truth, type="classic", verbose=True):
    """
    estimate the cost function
    """
    if verbose:
        str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(p)
        print("estimating the fonctionnal - p is {}".format(str))

    if type=="classic":
        x=RCpredict_Euler(step,p,x0,inputs)
    elif type=="krank":
        x=RCpredict_Krank(step,p,x0,inputs)
    return 0.5*np.sum(np.square(x-truth))/x.shape[0]

def RCgrad(step, p, x0, inputs, truth, verbose=True):
    """
    estimate the gradient with the Euler explicit scheme
    """
    n_par=len(p)
    n=x0.shape[0]
    if verbose:
        str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(p)
        print("estimating the gradient - p is {}".format(str))
    A, B, dA, dB = MatriX(p,jac=True)
    x=RCpredict_Euler(step,p,x0,inputs,allStates=True)

    z=np.zeros((n,n_par))
    df=np.zeros(n_par)

    for i in range(len(x)-1):
        for j in range(n_par):
            z[:,j]=np.linalg.inv(np.eye(2)-step*A).dot(z[:,j] + step*dA[j].dot(x[i+1]) + step*dB[j].dot(inputs[i]))
            df[j]+=z[0,j]*(x[i,0]-truth[i])

    return df/len(x)

def RCgrad_Krank(step, p, x0, inputs, truth, verbose=True):
    """
    estimate the gradient with the krank nicholson scheme
    """
    n_par=len(p)
    n=x0.shape[0]
    if verbose:
        str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(p)
        print("estimating the gradient - p is {}".format(str))
    A, B, dA, dB = MatriX(p,jac=True)
    x=RCpredict_Krank(step,p,x0,inputs,allStates=True)

    AS_B=np.linalg.inv(np.eye(n)-step*A/2)
    AS_C=AS_B.dot(np.eye(n)+step*A/2)
    AS_B=step*AS_B/2

    S=np.zeros((len(x),n))
    for j in range(len(x)-1,1,-1):
        S[j-1]=S[j].T.dot(AS_C)+np.array([x[j,0]-truth[j],0]).T.dot(AS_B)*2/len(x)

    df=np.zeros(n_par)
    for i in range(len(x)-1):
        for j in range(n_par):
            df[j]+=S[i].dot(dA[j].dot(x[i+1]+x[i])+dB[j].dot(inputs[i+1,:]+inputs[i,:]))/2
    return df


class RC_model():
    """
    tiny class to conduct the electrical modelization of a building
    """
    def __init__(self, house, params, nbptinh, p0, w0, FINASun=False):
        self._house = house
        self._params = params
        self._FINASun = FINASun
        if not FINASun:
            self._params.append({"name":"solar power (W)","color":"yellow"})
        self._step = 3600//nbptinh
        self._nbptinh = nbptinh
        self._p0 = p0
        self._w0 = w0
        self._wopt = []
        self._teta = []
        self._inputs = []
        self._truth = []
        self._algo="krank"
        self._exploreMatrix=[]

    def algo(self,algo):
        """
        krank or classic

        default is krank
        """
        self._algo = algo

    def setWeigths(self,p0,w0):
        self._p0 = p0
        self._w0 = w0

    def buildSet(self,smpStart,tDays,uid,tid,dir=dir):
        """
        :smpStart: unixtimestamp at which the sampling must start

        :tDays: number of days of sampling to consider

        the method will create a teta tensor gathering all the PHPFina feeds

        each column is a feed

        :uid: array of column indexes to construct the sollicitations tensor from teta

        :tid: column index to construct the truth vector from teta

        tid can be an array of 2 column indexes.

        In that case, the truth will be the average (through axis 1) of the corresponding teta columns between the 2 provided indexes
        """
        # fetching the feeds
        if self._FINASun:
            teta=GoToTensor(self._params,self._step,smpStart,tDays*24*self._nbptinh,dir=dir,FINASun=True)
        else:
            teta=GoToTensor(self._params[:-1],self._step,smpStart,tDays*24*self._nbptinh,dir=dir)
            # generate some sun
            # calculating the starting hour for the datarange
            smpH=datetime.utcfromtimestamp(smpStart).hour
            teta[:,-1]=generateSunRange(500,self._nbptinh, teta.shape[0], smpH)

        self._teta.append(teta)

        # adding a new sollicitations tensor
        inputs=[]
        for i in range(len(uid)):
            inputs.append(teta[:,uid[i]])
        self._inputs.append(np.vstack(inputs).T)

        # adding a new truth vector
        if isinstance(tid,list):
            self._truth.append(np.mean(teta[:,tid[0]:tid[1]],axis=1))
        if isinstance(tid,int):
            self._truth.append(teta[:,tid])

    def viewSet(self,i,guess,full=True):
        """
        this method will permit us to visualize a specific set
         and to make a prediction according to the current discretization scheme

        :i: the set number we want to vizualize

        :guess: initial values (ie at the set start) for (T_int,T_env)

        :full: False > only show the truth - True > show all the indoor temperature fields integrated to the set
        """
        if len(self._wopt):
            s=self._wopt*self._p0
        else:
            s=self._w0*self._p0

        if self._algo=="classic":
            T_sim=RCpredict_Euler(self._step, s, guess, self._inputs[i], allStates=True)
        if self._algo=="krank":
            T_sim=RCpredict_Krank(self._step, s, guess, self._inputs[i], allStates=True)

        if full:
            visualize(self._teta[i],self._params,self._house,Tint_sim=T_sim[:,0],TS_sim=T_sim[:,1],truth=self._truth[i])
        else:
            visualize(self._inputs[i],self._params,self._house,Tint_sim=T_sim[:,0],TS_sim=T_sim[:,1],truth=self._truth[i])


    def explorationMatrix(self,plan,verbose=True):
        """
        this method produces an exploration matrix for the parameters

        each line of the matrix is a parameters set

        :plan: array of the scenarios

        a scenario focuses on varying a single parameter only

        to define a scenario, you have to fix :

        - size

        - increment st

        - method (go, gof, no, rst)

        - parameter number

        go/rst : the parameter varies from st, 2*st, ..... size*st

        gof : same but the variation starts at snapshot[parameter_number]+st, NOT at st

        no : the parameter does not vary

        for go and gof, the snapshot is updated at the end of the scenario with last parameter value

        nothing is updated with the rst method

        """
        size=plan[0]
        st=plan[1]
        scenarios=plan[2]
        nb_par=plan[3]

        snapshot=copy.deepcopy(self._w0)

        n_par=self._w0.shape[0]
        n_li=np.sum(np.array(size))
        if verbose:
          print("total simulations to be achieved: {}".format(n_li))
        w=np.zeros((n_li,n_par))

        index=0
        # we loop on the scenarios
        for scenario in range(len(size)):
            # we fetch the parameter number to be explored
            j=nb_par[scenario]
            if verbose:
                print("going to simulate {} curves".format(size[scenario]))
            for i in range(size[scenario]):
                for k in range(n_par):
                    # the scenario has to explore the parameter
                    # unless the scenario explicitly says no
                    if k==j:
                        if scenarios[scenario]=="no":
                            w[index,k]=snapshot[k]
                        elif scenarios[scenario]=="gof":
                            w[index,k]=snapshot[k]+(i+1)*st[scenario]
                        else:
                            w[index,k]=(i+1)*st[scenario]
                    # the parameters are not supposed to evolve in the scenario
                    else:
                        if index==0:
                            w[index,k]=snapshot[k]
                        elif scenarios[scenario] in ["rst","no"] :
                            w[index,k]=snapshot[k]
                        else:
                            w[index,k]=w[index-1,k]
                if verbose:
                    print("index {} we have {}".format(index,w[index,:]))
                if index < n_li-1 :
                    index+=1
            if verbose:
                str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(w[index-1,:])
                print("we have reached {}".format(str))
            # we take a snapshot of the explored parameter unless the scenario is no or reset
            if scenarios[scenario] not in ["rst","no"]:
                snapshot[j]=w[index-1,j]
                #snapshot=w[index-1,:]
        self._exploreMatrix=w
        return w

    def explore(self,j,guess):
        """
        animation viewer

        :j: the set number - in order to work on _inputs[j] and _truth[j]

        :guess: initial values (ie at the set start) for (T_int,T_env)
        """
        # a small nested function to sequence the animation
        def animate(i):
            s=self._exploreMatrix[i,:]*self._p0
            str="%.2E, %.2E, %.2E, %.2E, %.2E" % tuple(s)
            T_sim=RCpredict_Krank(self._step, s, guess, self._inputs[j], allStates=True)
            xrange=np.arange(self._inputs[j].shape[0])
            tint.set_data(xrange,T_sim[:,0])
            #tint.set_color("yellow")
            tenv.set_data(xrange,T_sim[:,1])
            #tenv.set_color("gray")
            time_text.set_text(str)
            return tint, tenv, time_text

        xrange=np.arange(self._inputs[j].shape[0])
        fig = plt.figure()
        tint, = plt.plot(xrange,np.zeros(xrange.shape[0]),label="simulated indoor",color="red")
        tenv, = plt.plot(xrange,np.zeros(xrange.shape[0]),label="simulated envelope",color="gray")

        plt.plot(self._truth[j],label="truth",color="orange")
        plt.plot(self._inputs[j][:,0],label="outdoor",color="blue")
        plt.legend(loc='upper right')

        ymin, ymax = plt.gca().get_ylim()
        print('ymin is {} and ymax is {}'.format(ymin,ymax))
        time_text = plt.text(0, ymin+1, '', fontsize=10)

        frames=self._exploreMatrix.shape[0]

        ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, interval=100, repeat=False)

        plt.show()

    def optimize(self,i,guess,verbose=True):
        """
        launch a BFGS optimization on set i
        """
        # we will use array w to store the evolution of the parameters during the iteration process
        # they will stand as quality indicators for convergence or not
        w=[]

        # nested functions for regularisation
        def fonc(_w):
            return RCfonc(self._step, self._p0*_w, guess, self._inputs[i], self._truth[i], type=self._algo, verbose=verbose)

        def grad(_w):
            w.append(_w)
            if self._algo=="krank":
                return self._p0*RCgrad_Krank(self._step, self._p0*_w, guess, self._inputs[i], self._truth[i], verbose=verbose)
            elif self._algo=="classic":
                return self._p0*RCgrad(self._step, self._p0*_w, guess, self._inputs[i], self._truth[i], verbose=verbose)

        res=optimize.minimize(fonc, self._w0, method="BFGS", jac=grad, options={"disp":False})
        #bounds=[(0,np.inf),(0,np.inf),(0,1),(0,1),(0,1)]
        #res=optimize.minimize(fonc, self._w0, method="L-BFGS-B", jac=grad, bounds=bounds)

        # SANITY CONVERGENCE CHECK
        quality=np.array(w)
        nb=321
        lib=["cres", "cs", "ri", "r0", "rf"]
        for z in range(len(lib)):
            str="%.0E" % (1/self._p0[z])
            lib[z]="{} x {}".format(lib[z],str)
        #it is the iteration number
        for it in range(quality.shape[-1]):
            plt.subplot(nb)
            plt.plot(quality[:,it],label=lib[it])
            plt.legend()
            nb+=1
        plt.show()

        print(res)
        popt=res["x"]*self._p0
        print(popt)
        self._wopt = res["x"]
