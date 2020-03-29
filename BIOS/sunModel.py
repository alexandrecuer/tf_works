import math
import numpy as np
import matplotlib.pylab as plt

"""
KASTEN MODEL FOR SUN POWER MODELISATION
work in progress
"""

# Escurolles is lat 46°8'39'' long 3°15'58''
# in decimal deg.
lat=46.1441667
# in decimal deg.
long=3.2661111
# in meter
alt=300

# solar constant in W/m2
I0=1367
# offset with the Universal Time Coordinated
UTC_offset=1

"""
for all the following methods :
n day number - 1th of january > n=1 ....
h : local time between 0 and 23
"""

def extraRadiation(n):
    """
    calculates the extraterrestrial radiative flux in W/m2
    """
    return I0*(1+0.0334*math.cos(2*math.pi*n/365))

def deltaT(n,hour=True):
    """
    time variation for day n in hour
    related to the earth orbit disturbance
    result in hour or minutes
    """
    B=2*math.pi*(n-1)/365
    E=229.2*(0.000075+0.001868*math.cos(B)-0.032077*math.sin(B)-0.014615*math.cos(2*B)-0.04089*math.sin(2*B))
    if hour==True:
        return E/60
    else:
        return E

def hourAngle(n,h,rad=True):
    """
    angle in radians between the sun and the local meridian
    result in degrees or radians
    """
    ha = 15*(h-UTC_offset+deltaT(n)-12) + long

    if rad==True:
        return math.radians(ha)
    else:
        return ha


def earthDeclination(n,rad=True):
    """
    angle between the equatorial plane and the line connecting earth and sun centres
    result in degrees or radians
    """
    ed=23.45*math.sin(2*math.pi*(284+n)/365)
    if rad==True:
        return math.radians(ed)
    else:
        return ed

def sunDuration(n):
    """
    calculates the sun duration for the day n
    result in hour
    """
    return 2*math.degrees(math.acos(-math.tan(math.radians(lat))*math.tan(earthDeclination(n))))/15

def solarAngles(n,h,rad=True):
    """
    calculates :
    - (SH)solar/angular height or solar altitude = angle between the local horizontal plane and the sun direction
    - (AZ)azimuth = angle between the local meridian and the vertical plane including the observed point and the sun
    results in degrees or radians
    """
    decl=earthDeclination(n)
    sinh = math.sin(math.radians(lat))*math.sin(decl) + math.cos(math.radians(lat))*math.cos(decl)*math.cos(hourAngle(n,h))
    SH=math.asin(sinh)
    AZ=math.asin(math.cos(decl)*math.sin(hourAngle(n,h)))/math.cos(SH)
    if SH < 0:
        SH=0

    if rad==True:
        return SH, AZ
    else:
        return math.degrees(SH), math.degrees(AZ)

def Linke(n,h):
    """
    gas absorption disorder (02, CO2, O3 ozone, water vapor, aerosols)
    Capderou ?
    """
    sinphi = math.sin(math.radians(lat))
    A=math.sin(2*math.pi*(n-121)/365)
    z=alt/1000
    angles=solarAngles(n,h)
    T1 = 2.4 - 0.9*sinphi + 0.1*(2 + sinphi*A - 0.2*z - (1.22+0.14*A)*(1-math.sin(angles[0])))
    T2 = 0.89**z
    T3 = (0.9 + 0.4*A)*0.63**z
    return T1+T2+T3

def globalRadiation(n,h):
    """
    in W/m2
    cf OMM
    https://hal.archives-ouvertes.fr/jpa-00246138/document
    """
    angles=solarAngles(n,h)
    TL=Linke(n,h)
    z=alt/1000
    mod1=(1300-57*TL)*math.exp(0.22*z/7.8)*math.sin(angles[0])**((TL+36)/33)
    mod2=extraRadiation(n)*math.sin(angles[0])
    return mod1, mod2

for i in [1,100,200,300]:
    print("day  duration of day {} is {} h".format(i, sunDuration(i)))

E=np.zeros(365)
delta=np.zeros(365)
for j in range(1,366):
    E[j-1]=deltaT(j,hour=False)
    delta[j-1]=earthDeclination(j,rad=False)

plt.subplot(111)
plt.plot(E,label="Time Equation in minutes")
plt.plot(delta,label="earth declination in degrees")
plt.xlabel("Time in day")
plt.legend()
plt.show()


lk=[]
gr=[]
for day in range(1,366,1):
    for h in range(24):
        lk.append(Linke(day,h))
        gr.append(globalRadiation(day,h)[0])

ax1=plt.subplot(111)
plt.xlabel("Time in hour")
plt.plot(gr,label="global radiation in W/m2",color="orange")
plt.legend(loc='upper left')
ax2 = ax1.twinx()
plt.plot(lk,label="Linke trouble",color="green")
plt.legend(loc='upper right')
plt.show()
