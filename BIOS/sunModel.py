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
        self._lat=lat
        self._radlat=math.radians(self._lat)
        self._long=long
        # altitude in meter
        self._alt=alt
        # offset with the Universal Time Coordinated
        self._UTCoffset=0
        self._nbptinh=nbptinh

        # evaluating the starting timestamp = same UTC day at 00:00:00
        tt=time.gmtime(start)
        sec_elapsed_since_UTCday_start=tt.tm_hour*3600+tt.tm_min*60+tt.tm_sec
        utsStart=start-sec_elapsed_since_UTCday_start
        tts=time.gmtime(utsStart)
        self._utsStart=utsStart
        print("you have entered the following start {} UTC".format(time.strftime('%Y-%m-%d %H:%M:%S',tt)))
        print("sun generation will start at unixtimestamp {} which is {} UTC".format(utsStart,time.strftime('%Y-%m-%d %H:%M:%S',tts)))
        print("this is UTC day number {} of the year".format(tts.tm_yday))

        self._startday = tts.tm_yday
        self._nbdays = nbdays
        self._indice=0
        self._datas=np.zeros((self._nbdays*24*self._nbptinh, 5))
        # Energy received on the period
        self._E=0

    def hourAngle(self,n,h,rad=True):
        """
        angle between the sun and the local meridian
        result in degrees or radians
        """
        w = 15*(h-self._UTCoffset+deltaT(n)-12) + self._long

        if rad==True:
            return math.radians(w)
        else:
            return w

    def sunDuration(self,n):
        """
        calculates the sun duration for the day n
        result in hour
        """
        return 2*math.degrees(math.acos(-math.tan(self._radlat)*math.tan(earthDeclination(n))))/15

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
        sh = math.sin(self._radlat)*math.sin(decl) + math.cos(self._radlat)*math.cos(decl)*math.cos(w)
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
        sinphi = math.sin(self._radlat)
        A=math.sin(2*math.pi*(n-121)/365)
        z=self._alt/1000
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
        z=self._alt/1000
        # solar gain / coefficient d'atténuation tenant compte des variations de la nébulosité
        fs=0.65
        fs=1
        mod1=fs*(1300-57*TL)*math.exp(0.22*z/7.8)*math.sin(angles[0])**((TL+36)/33)
        mod2=fs*extraRadiation(n)*math.sin(angles[0])
        self._datas[self._indice]=np.array([mod1,TL,math.degrees(angles[0]),math.degrees(angles[1]),math.degrees(angles[2])])
        self._indice+=1
        return mod1, mod2

    def generate(self):
        """
        generate and store datas
        global radiation in W/m2, linke trouble, gamma, alpha, omega
        """
        for d in range(self._nbdays):
            day=self._startday+d
            for h in range(24):
                for m in range(self._nbptinh):
                    self.globalRadiation(day,h+m/self._nbptinh)

    def energy(self):
        E=[]
        steps_in_month=30*24*self._nbptinh
        indice=0
        Emonth=0
        for i in range(self._datas.shape[0]):
            if indice > steps_in_month:
                indice=0
                E.append(Emonth)
                Emonth=0
            Emonth+=self._datas[i,0]
            indice+=1
        if len(E):
            print("Kwh not integrated in the mensual estimation : {}".format(Emonth/(self._nbptinh*1000)))
            E=np.array(E)/(self._nbptinh*1000)
            print("the following array synthetises mensual estimations on the choosen period")
            print(E)
            self._E=np.sum(E)
        else:
            self._E=Emonth/(self._nbptinh*1000)
