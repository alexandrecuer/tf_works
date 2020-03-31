import math
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
from datetime import datetime
import time
print("your computer is UTC+{} but we are going to work in UTC".format(-time.timezone//3600))

"""
KASTEN MODEL FOR GLOBAL SUN RADIARION MODELISATION
CLEAR SKY ONLY

online ressources on solar irradiation : https://ec.europa.eu/jrc/en/pvgis

work in progress

for all the following methods :
n day number - 1th of january > n=1 ....
h : local time in hour
"""

def viewSunPath():
    """
    visualize the sun path for a classic day on planet earth
    """
    # observation point
    px = 4
    py = 3
    pz = 0
    r = 3

    # parallel or circle of latitude
    x0 = [px, px]
    y0 = [0, 6]
    z0 = [0, 0]

    # meridian
    x1 = [0, 6]
    y1 = [py, py]
    z1 = [0, 0]

    # sun's path during the day
    theta=np.linspace(-np.pi/2,np.pi/4,600)
    c1x=r/2+np.zeros(600)
    c1y=py+r*np.sin(theta)
    c1z=r*np.cos(theta)

    xw = r/2
    yw = py
    zw = r+0.15

    theta2=np.linspace(np.pi/4,np.pi/2,200)
    c2x=r/2+np.zeros(200)
    c2y=py+r*np.sin(theta2)
    c2z=r*np.cos(theta2)

    # sun position
    sunx=r/2
    suny=py+r*np.cos(np.pi/4)
    sunz=r*np.sin(np.pi/4)

    # sun ray
    rayx=[px,sunx]
    rayy=[py,suny]
    rayz=[pz,sunz]

    # gamma
    d=((sunx-px)**2+(suny-py)**2+(sunz-pz)**2)**0.5
    dsina=sunz
    gamma=math.asin(dsina/d)
    u = np.linspace(0, gamma, 50)
    p = ((sunx-px)**2 + (suny-py)**2)**0.5
    v = math.acos((px-sunx)/p)
    zarcg = pz + d/4 * np.sin(u)
    xarcg = px - d/4 * np.cos(u) * np.cos(v)
    yarcg = py + d/4 * np.cos(u) * np.sin(v)

    zg = pz + d/4 * np.sin(gamma/2)
    xg = px - d/4 * np.cos(gamma/2) * np.cos(v)
    yg = py + d/4 * np.cos(gamma/2) * np.sin(v)

    # alpha
    alpha=np.linspace(0, v, 50)
    xarca = px - (px - r/2) * np.cos(alpha)
    yarca = py + (px - r/2) * np.sin(alpha)
    zarca = np.zeros(50)

    xa = px - (px - r/2 + 0.15) * np.cos(v/2)
    ya = py + (px - r/2 + 0.15) * np.sin(v/2)
    za = 0

    # Plotting....
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title("vizualisation of the sun's path")
    ax.plot(xarca, yarca, zarca, color="blue")
    ax.text(xa,ya,za,"α", fontsize=18, color="blue")
    ax.plot(xarcg, yarcg, zarcg, color="blue")
    ax.text(xg,yg,zg,"$\gamma$", fontsize=18, color="blue")

    ax.text(px, 0, 0.1 , "E", fontsize=12, color="gray")
    ax.text(px, 6, 0.1 , "W", fontsize=12, color="gray")
    ax.plot(x0, y0, z0, '--', color="gray", label = "circle of latitude")
    ax.text(0, py, 0.1 , "S", fontsize=12, color="green")
    ax.text(6, py, 0.1 , "N", fontsize=12, color="green")
    ax.plot(x1, y1, z1, '--', color="green", label = "meridian")

    ax.plot(c1x, c1y, c1z, color="blue")
    ax.plot(c2x, c2y, c2z, '--', color="blue")
    ax.text(xw,yw,zw,"w ($\omega$)", fontsize=12, color="blue")

    ax.plot(rayx, rayy, rayz, color="orange")

    # observation point
    ax.scatter(px,py,pz,marker='o',color="black")
    ax.text(px, py, 0.1 , "P", fontsize=10)

    # sun at sunrise and sunset
    ax.scatter(sunx,0,0,marker='o',s=56,color="orange",label="sunrise",alpha=0.4)
    ax.scatter(sunx,2*py,0,marker='o',s=56,color="red",label="sunrset",alpha=0.4)
    # sun at actual position
    ax.scatter(sunx,suny,sunz,marker='o',s=56,color="yellow")
    #projections for gamma vizualisation
    ax.plot([sunx,sunx],[suny,suny],[sunz,0], '--', color="orange")
    ax.plot([px,sunx],[py,suny],[pz,0], '--', color="orange")

    ax.legend(loc='lower left')

    plt.show()

def extraRadiation(n):
    """
    calculates the extraterrestrial radiative flux in W/m2
    """
    # solar constant in W/m2
    I0=1367
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

class globalSunRadiation():
    def __init__(self,lat,long,alt,nbptinh,start,nbdays):
        # latitude and longitude in decimal deg.
        self.lat=lat
        self.radlat=math.radians(self.lat)
        self.long=long
        # altitude in meter
        self.alt=alt
        # offset with the Universal Time Coordinated
        self.UTCoffset=0
        self.nbptinh=nbptinh
        #Return the UTC datetime from the unix timestamp
        human=datetime.utcfromtimestamp(start)
        # time tuple is tm_year, tm_mon, tm_mday, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst=-1 (day saving time)
        tt=human.timetuple()
        print("you have choosen to generate global sun rad. on {} which is day of the year number {}".format(human,tt.tm_yday))
        self.startday = tt.tm_yday
        self.nbdays = nbdays
        self.indice=0
        self.datas=np.zeros((self.nbdays*24*self.nbptinh, 5))
        # Energy received on the period
        self.E=0

    def hourAngle(self,n,h,rad=True):
        """
        angle between the sun and the local meridian
        result in degrees or radians
        """
        w = 15*(h-self.UTCoffset+deltaT(n)-12) + self.long

        if rad==True:
            return math.radians(w)
        else:
            return w

    def sunDuration(self,n):
        """
        calculates the sun duration for the day n
        result in hour
        """
        return 2*math.degrees(math.acos(-math.tan(self.radlat)*math.tan(earthDeclination(n))))/15

    def solarAngles(self,n,h,rad=True):
        """
        calculates the solar angles :
        - (gamma)solar/angular height or solar altitude = angle between the local horizontal plane and the sun direction
        - (alpha)azimuth = angle between the local meridian and the vertical plane including the observed point and the sun
        results in degrees or radians
        """
        decl=earthDeclination(n)
        w=self.hourAngle(n,h)
        #hauteur du soleil - sun height : sh
        sh = math.sin(self.radlat)*math.sin(decl) + math.cos(self.radlat)*math.cos(decl)*math.cos(w)
        gamma=math.asin(sh)
        #trace du soleil dans le plan horizontal
        alpha=math.asin(math.cos(decl)*math.sin(w)/math.cos(gamma))
        if gamma < 0:
            gamma=0

        if rad==True:
            return gamma, alpha, w
        else:
            return math.degrees(gamma), math.degrees(alpha), math.degrees(w)

    def Linke(self,n,h):
        """
        gas absorption disorder (02, CO2, O3 ozone, water vapor, aerosols)
        Capderou ?
        """
        sinphi = math.sin(self.radlat)
        A=math.sin(2*math.pi*(n-121)/365)
        z=self.alt/1000
        angles=self.solarAngles(n,h)
        T1 = 2.4 - 0.9*sinphi + 0.1*(2 + sinphi*A - 0.2*z - (1.22+0.14*A)*(1-math.sin(angles[0])))
        T2 = 0.89**z
        T3 = (0.9 + 0.4*A)*0.63**z
        return T1+T2+T3

    def globalRadiation(self,n,h):
        """
        in W/m2
        cf OMM
        https://hal.archives-ouvertes.fr/jpa-00246138/document
        """
        angles=self.solarAngles(n,h)
        TL=self.Linke(n,h)
        z=alt/1000
        # solar gain / coefficient d'atténuation tenant compte des variations de la nébulosité
        fs=0.65
        mod1=fs*(1300-57*TL)*math.exp(0.22*z/7.8)*math.sin(angles[0])**((TL+36)/33)
        mod2=fs*extraRadiation(n)*math.sin(angles[0])
        self.datas[self.indice]=np.array([mod1,TL,math.degrees(angles[0]),math.degrees(angles[1]),math.degrees(angles[2])])
        self.indice+=1
        return mod1, mod2

    def generate(self):
        """
        generate and store datas
        global radiation in W/m2, linke trouble, gamma, alpha, omega
        """
        for d in range(self.nbdays):
            day=self.startday+d
            for h in range(24):
                for m in range(self.nbptinh):
                    self.globalRadiation(day,h+m/nbptinh)

    def energy(self):
        E=[]
        steps_in_month=30*24*nbptinh
        indice=0
        Emonth=0
        for i in range(self.datas.shape[0]):
            if indice > steps_in_month:
                indice=0
                E.append(Emonth)
                Emonth=0
            Emonth+=self.datas[i,0]
            indice+=1
        if len(E):
            print("Kwh not integrated in the mensual estimation : {}".format(Emonth/(self.nbptinh*1000)))
            E=np.array(E)/(self.nbptinh*1000)
            print("the following array synthetises mensual estimations on the choosen period")
            print(E)
            self.E=np.sum(E)
        else:
            self.E=Emonth/(self.nbptinh*1000)

# Escurolles is lat 46°8'39'' long 3°15'58''
# in decimal deg.
lat=46.1441667
# in decimal deg.
long=3.2661111
# in meter
alt=300

# all this is UTC
# january 10 2019, 12:00
smpStart=1547121600
# october 20 2018, 0:00
#smpStart=1539993600
tDays=15

#december 2018
smpStart=1543618740
tDays=30

#august 2018
smpStart=1533081600
tDays=30

#all 2018
smpStart=1514764800
tDays=365

nbptinh=12

sun=globalSunRadiation(lat,long,alt,nbptinh,smpStart,tDays)
"""
for i in [1,100,200,300]:
    print("day  duration of day {} is {} h".format(i, sun.sunDuration(i)))
"""
sun.generate()
sun.energy()
print(sun.E)



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

viewSunPath()

ax1=plt.subplot(211)
plt.plot(sun.datas[:,0],label="global radiation in W/m2",color="orange")
plt.legend(loc='upper left')
ax2 = ax1.twinx()
plt.plot(sun.datas[:,1],label="Linke trouble",color="green")
plt.legend(loc='upper right')
plt.subplot(212,sharex=ax1)
plt.plot(sun.datas[:,2],label="gamma-height",color="orange")
plt.plot(sun.datas[:,3],label="alpha-zenith",color="red")
plt.plot(sun.datas[:,4],label="omega-hour angle",color="pink")
plt.legend(loc='upper right')
plt.xlabel("Time - step=h/{}".format(nbptinh))
plt.show()
