from src.modelrc import RCpredict_Krank, MatriX
from src.tools import PHPFina,getMetas
import numpy as np
import math
import matplotlib.pylab as plt
import copy
#import seaborn

verbose=False

"""
parameters resulting from the gradient optimisation with the src.modelrc module
"""
p=[1.58e+09, 2.16e+10, 1.96e-04, 2.12e-01, 3.546e-04]

Ti0=14
# température de confort
Tc=20

step=1800
nbDays=60
nb=nbDays*24*3600//step
#start=1577404800
start=1577874830
from datetime import datetime
import time
from dateutil import tz
CET=tz.gettz('Europe/Paris')

def tsToTuple(ts):
    _time=datetime.fromtimestamp(ts,CET)
    _tuple=_time.timetuple()
    # return a tuple tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst
    return(_tuple)

dir="../../labo/phpfina"

"""
create a numpy array from a FINA feed
"""
def FinaToNp(feed,nb):
    feed.getMetas()
    feed.setStart(start)
    feed.getDatas(nb)
    feed._datas=np.asarray(feed._datas)

"""
outdoor temp is feed 18
sun radiation is feed 21
hot water (dep) is 25
hot water (return) is 29
power is 66
consigne Batisense : 57
"""
Text=PHPFina(18,step,dir)
FinaToNp(Text,nb)
"""
Tw=PHPFina(25,step,dir)
FinaToNp(Tw,nb)
"""
Sun=PHPFina(21,step,dir)
FinaToNp(Sun,nb)
Power=PHPFina(66,step,dir)
FinaToNp(Power,nb)
Batisense=PHPFina(57,step,dir)
FinaToNp(Batisense,nb)

"""
#signature énergétique du bâtiment ? ? ?
#pas très convaincant
#puissance consommée en W v température extérieure
#nota : utilise module seaborn
plt.subplot(211)
plt.ylabel("Power W")
plt.plot(Text._datas,Power._datas,'.')
plt.subplot(212)
plt.xlabel("Text °C")
seaborn.kdeplot(Text._datas,Power._datas, cmap="Reds", shade=True, shade_lowest=False)
plt.show()
"""

# flow rate in m3/h
flow_rate = 5.19
deltaT=20
Cw=1162.5 #Wh/m3/K

pwMax=deltaT*flow_rate*Cw

"""
introduction of all matriX tools necessary for the models
"""
A, B = MatriX(p,jac=False)
n=A.shape[0]
AS_B=np.linalg.inv(np.eye(n)-step*A/2)
AS_C=AS_B.dot(np.eye(n)+step*A/2)
AS_B=step*AS_B/2


def predict_Euler(x,inputs):
    """
    predict the next point using an Euler scheme

    inputs : np.array([ Text(s), Phea(s), Sun(s) ])

    x      : np.array([ Tint(s), Tsurface(s) ])

    output : np.array([ Tint(s+1), Tsurface(s+1) ])
    """
    return np.linalg.inv(np.eye(n)-step*A).dot(x+step*B.dot(inputs))

def predict_Krank(x,inputs):
    """
    predict the next point using a Krank-Nicholson scheme

    you need to give sollicitations both at the step and next step

    inputs : np.array([ [ Text(s)  ,Phea(s)  ,Sun(s)   ], [ Text(s+1),Phea(s+1),Sun(s+1) ] ])

    x      : np.array([ Tint(s), Tsurface(s) ])

    output : np.array([ Tint(s+1), Tsurface(s+1) ])
    """
    return AS_C.dot(x)+AS_B.dot(B.dot(inputs[1]+inputs[0]))

def Ts(ti,te):
    """
    evaluate the envelope temperature given indoor and outdoor temperature

    ONLY USED AS A STARTING POINT

    simulation outputs indoor AND envelope temperatures
    """
    return (ti+2*te)/3


"""
datas Tensor
column 0 : outdoor temperature
column 1 : heating power in W
column 2 : global sun radiation
column 3 : indoor temperature > return from model (sensor)
column 4 : envelope temperature > return from model (will not be available on the field)
"""
datas=np.zeros((Text._datas.shape[0],5))
datas[:,0]=Text._datas
datas[:,2]=50*Sun._datas
datas[0,3]=Ti0
datas[0,4]=Ts(datas[0,3],datas[0,0])
#print(datas)

"""
building an agenda indicating weather people are working or not
1: work
0: rest
here we go for fixed working hours each day
start at 8 and stop at 17
"""
agenda=np.zeros(datas.shape[0])
time=start
tpl=tsToTuple(time)
work=0
if tpl.tm_hour in range(8,17):
    if tpl.tm_wday not in [5,6]:
        work=1
agenda[0]=work
for i in range (0,datas.shape[0]-1):
    tpl=tsToTuple(time)
    if tpl.tm_hour==17 and previous.tm_hour==16:
        if tpl.tm_wday not in [5,6]:
            work=0
    if tpl.tm_hour==8 and previous.tm_hour==7:
        if tpl.tm_wday not in [5,6]:
            work=1
    agenda[i]=work
    previous=tpl
    time+=step
plt.subplot(111)
plt.plot(agenda, label="agenda de présence")
plt.legend()
plt.show()

def GetLevelDuration(i):
    """
    return the supposed duration of the level in number of steps
    a level = period during which we can see no change in the agenda
    """
    j=i
    while(agenda[j]==agenda[j+1]):
        if j < datas.shape[0]-2:
            j+=1
        else:
            break
    return j+1-i

def monitor(i,field_mode=False):
    """
    this method simulates the monitoring by a sensor and feed the datas tensor - datas[i,3:5]
    field_mode=False > uses the model
    """
    if not field_mode:
        x=datas[i-1,3:5]
        inputs=datas[i-1,0:3]
        datas[i,3:5]=predict_Euler(x,inputs)

def hysteresis(Tc=Tc):
    """
    closed loop with a simple hysteresis
    """
    K=15
    now=start
    heating=True

    for i in range (0,datas.shape[0]-1):

        """
        1) check where we are in the agenda
        update the value of the boolean heating var
        """
        if i > 0:
            if agenda[i]!=agenda[i-1]:
                heating = not heating
                if heating==False:
                    goto=GetLevelDuration(i)
        """
        2) if we are not in heating mode, we conduct a simulation up to the target
        """
        if heating==False:
            inputs=copy.deepcopy(datas[i:i+goto,:3])
            inputs[:,1]=np.ones(inputs.shape[0])*pwMax
            x0=np.array( [ datas[i,3], Ts(datas[i,3],datas[i,0]) ] )
            T=RCpredict_Krank(step,p,x0,inputs,allStates=True)
            if verbose:
                print("we are at step {}".format(i))
                print("indoor temp. is now {}".format(datas[i,3]))
                print("{} remaining points before reaching target - if starting heating now, indoor temp. at target will be {}".format(goto,T[-1,0]))
            if T[-1,0]<Tc:
                heating=True
            goto-=1

        """
        3) the controller = hysteresis
        if we are in heating mode, we check if temperature is above Tconfort
        if so we do not heat
        introduction d'une sorte de régulation proportionnelle fonction de l'erreur ???
        erreur = écart entre la température intérieure monitorée et la consigne de confort
        """
        if datas[i,3] < Tc and heating==True:
            e=Tc-datas[i,3]
            datas[i,1]=min(K*e*pwMax,pwMax)

        """
        4) monitoring to update the datas tensor
        """
        now+=step
        monitor(i+1)

hysteresis()

xrange=np.arange(Sun._datas.shape[0])
ax1=plt.subplot(111)
plt.plot(datas[:,0], label="Text")
plt.plot(datas[:,3], label="Tint")
plt.plot(datas[:,4], label="Ts")
plt.fill_between(xrange,Tc*agenda,color="green",alpha=0.4,label="consigne °C")
plt.legend(loc='upper left')
ax2=ax1.twinx()
ax2.set_ylim(0, 250000)
plt.fill_between(xrange,datas[:,2], label="sun radiation W", color="yellow",alpha=0.4)
plt.plot(datas[:,1], color="red", label="power W")
plt.legend(loc='upper right')
plt.show()

Ekwh=np.sum(datas[:,1])*step/(3600*1000)
Bkwh=np.sum(Power._datas*step/(3600*1000))

ax1=plt.subplot(111)
plt.title("conso Batisense {} (KWh) - conso RC Hysteresys {} (Kwh)".format("%.2E" % Bkwh,"%.2E" % Ekwh))
plt.plot(datas[:,1], color="orange",label="power consumed - Cerema Hysteresis model")
plt.plot(Power._datas, color="red",label="power consumed with Batisense decision")
plt.legend(loc='upper left')
ax2=ax1.twinx()
ax2.set_ylim(0, 20)
plt.plot(Batisense._datas, color="purple", label="batisense On Off")
plt.legend(loc='upper right')
plt.show()

"""
piste 2 à creuser ?
changer le contrôleur hysteresis pour un contrôleur lois d'eau pour calculer la puissance utilisée
cf http://www.hbsoft.be/chauffage/radiateur.html
celà nécessite de connaitre le circuit avec un peu de détail
nombre de radiateurs, estimation des pertes....
pour le bâtiment du labo de Clermont :
- circuit cellules : 28 radiateurs
- circuit nord : 34 radiateurs
- circuit sud : 31 radiateurs
"""
def waterLaw(xa,ya,xb,yb):
    """
    a simple water law for the circuit
    """
    a=(yb-ya)/(xb-xa)
    b=ya-a*xa
    return a, b

a_cells, b_cells = waterLaw(-10,85,20,40)

print ("Loi d'eau du circuit - pente {} ordonnée à l'origine {}".format(a_cells, b_cells))

Tw_target=a_cells*Text._datas+b_cells
