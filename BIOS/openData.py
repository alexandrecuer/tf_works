import os
import math
import numpy as np
import time
import datetime
from matplotlib import pyplot as plt
import copy
import requests
# cf https://www.urlencoder.io/python/
import urllib.parse
from dateutil import tz

def ODSstrToUTS(str):
    # removing the last occurence of : in the time string
    # see https://docs.python.org/fr/3.6/library/datetime.html#strftime-strptime-behavior
    # and https://stackoverflow.com/questions/12281975/convert-timestamps-with-offset-to-datetime-obj-using-strptime
    tstr=str[::-1].replace(":","",1)[::-1]
    _time=datetime.datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S%z')
    ts=int(datetime.datetime.timestamp(_time))
    return ts

class openData():
    """
    to get the summary of the dataset and retrieve all the fields codes/descriptions

    https://data.opendatasoft.com/api/v2/opendatasoft/datasets/donnees-synop-essentielles-omm%40public

    numer_sta : ID OMM station
    date : date
    pmer : Pression au niveau mer
    tend : Variation de pression en 3 heures(Pa)
    cod_tend : Type de tendance barométrique
    dd : Direction du vent moyen 10 mn (°)
    ff : Vitesse du vent moyen 10 mn (m/s)
    t : Température(K)
    td : Point de rosée(K)
    u : Humidité(%)
    vv : Visibilité horizontale(m)
    ww : Temps présent
    w1 : Temps passé 1
    w2 : Temps passé 2
    n : Nebulosité totale(%)
    nbas : Nébulosité  des nuages de l' étage inférieur(octa)
    hbas : Hauteur de la base des nuages de l'étage inférieur(mètre)
    cl : Type des nuages de l'étage inférieur
    cm : Type des nuages de l'étage moyen
    ch : Type des nuages de l'étage supérieur
    pres : Pression station(Pa)
    niv_bar : Niveau barométrique(Pa)
    geop : Géopotentiel(m2/s2)
    tend24 : Variation de pression en 24 heures(Pa)
    tn12 : Température minimale sur 12 heures(K)
    tn24 : Température minimale sur 24 heures(K)
    tx12 : Température maximale sur 12 heures(K)
    tx24 : Température maximale sur 24 heures(K)
    tminsol : Température minimale du sol sur 12 heures(K)
    sw : Méthode de mesure Température du thermomètre mouillé
    tw : Température du thermomètre mouillé(K)
    raf10 : Rafale sur les 10 dernières minutes(m/s)
    rafper : Rafales sur une période(m/s)
    per : Periode de mesure de la rafale(min)
    etat_sol : Etat du sol
    ht_neige : Hauteur totale de la couche de neige, glace, autre au sol(m)
    ssfrai : Hauteur de la neige fraîche(m)
    perssfrai : Periode de mesure de la neige fraiche(1/10 heure)
    rr1 : Précipitations dans la dernière heure(mm)
    rr3 : Précipitations dans les 3 dernières heures(mm)
    rr6 : Précipitations dans les 6 dernières heures(mm)
    rr12 : Précipitations dans les 12 dernières heures(mm)
    rr24 : Précipitations dans les 24 dernières heures(mm)
    phenspe1 : Phénomène spécial 1
    phenspe2 : Phénomène spécial 2
    phenspe3 : Phénomène spécial 3
    phenspe4 : Phénomène spécial 4
    nnuage1 : Nébulosité couche nuageuse 1(octa)
    ctype1 : Type nuage 1
    hnuage1 : Hauteur de base 1(m)
    nnuage2 : Nébulosité couche nuageuse 2(octa)
    ctype2 : Type nuage 2
    hnuage2 : Hauteur de base 2(m)
    nnuage3 : Nébulosité couche nuageuse 3(octa)
    ctype3 : Type nuage 3
    hnuage3 : Hauteur de base 3(m)
    nnuage4 : Nébulosité couche nuageuse 4(octa)
    ctype4 : Type nuage 4
    hnuage4 : Hauteur de base 4(m)
    coordonnees : Coordonnees
    nom : Nom
    type_de_tendance_barometrique : Type de tendance barométrique
    temps_passe_1 : Temps passé 1
    temps_present : Temps présent
    tc : Température (°C)
    tn12c : Température minimale sur 12 heures (°C)
    tn24c : Température minimale sur 24 heures (°C)
    tx12c : Température maximale sur 12 heures (°C)
    tx24c : Température maximale sur 24 heures (°C)
    tminsolc : Température minimale du sol sur 12 heures (en °C)
    altitude : Altitude
    longitude : Longitude
    latitude : Latitude
    libgeo : communes (name)
    codegeo : communes (code)
    nom_epci : EPCI (name)
    code_epci : EPCI (code)
    nom_dept : department (name)
    code_dep : department (code)
    nom_reg : region (name)
    code_reg : region (code)
    mois_de_l_annee : mois_de_l_annee
    """
    def __init__(self,dataset,station,start,stop,fields,utz,step_in_h,year=True):
        """
        using the API v2 server
        API v1 was "https://data.opendatasoft.com/explore/dataset"
        start and stop can be integer if working with years
        if not, they must be string ODS formatted
        """
        self._server="https://data.opendatasoft.com/api/v2/opendatasoft/datasets"
        self._dataset=dataset
        self._station=station
        self._start=start
        self._stop=stop
        self._nbf=len(fields)
        self._fields=','.join(fields)
        self._tz=utz
        self._step_in_h=step_in_h
        self._step_in_s=step_in_h*3600
        if year:
            _time=datetime.datetime(self._start,1,1,tzinfo=tz.gettz(utz))
            self._uts=int(datetime.datetime.timestamp(_time))
            self._nbp=(self._stop-self._start)*365*24//self._step_in_h
        else:
            self._uts=ODSstrToUTS(start)
            self._nbp=(ODSstrToUTS(stop)-self._uts)//(3600*self._step_in_h)
        self._full_data=np.zeros((self._nbp,self._nbf))

    def retrieve(self,view={'vis':False}):
        params={
                 'where':['numer_sta="{}"'.format(self._station),'date<\'{}\''.format(self._stop),'date>=\'{}\''.format(self._start)],
                 'sort':'date',
                 'select': self._fields,
                 'timezone':self._tz,
                 'delimiter':';'
               }
        urlend=urllib.parse.urlencode(params,safe='',doseq=True)
        url="{}/{}/exports/csv?{}".format(self._server,self._dataset,urlend)
        #print(url)
        #input("press any key")

        response = requests.get(url)
        data=response.text
        data = data.rstrip('\n\r')
        lines = data.split('\n')
        header = lines[0].split(';')
        lines = lines[1:]
        print("we've got {} lines and {} columns".format(len(lines), len(header)))
        # raw_data shape is (time,features)
        raw_data=np.zeros((len(lines),self._nbf))
        missing=0
        for i,line in enumerate(lines):
            x = line.split(';')

            #time_str=x[0][::-1].replace(":","",1)[::-1]
            # converting to unixtimestamp
            raw_data[i,0]=ODSstrToUTS(x[0])
            #time.mktime(datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S%z').timetuple())
            for j in range(1,self._nbf,1):
                if x[j]:
                  raw_data[i,j]=float(x[j])
                else:
                  raw_data[i,j]= math.nan
                  missing+=1

        print("according to the presumed time stamp, we should have {} points".format(self._nbp))
        print("missing datas : {}".format(missing))

        #remove eventual lines full of zeros
        raw_data = raw_data[~np.all(raw_data == 0, axis=1)]

        # reorder by ascending timestep
        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        # datas are supposed to be sorted by ascending date but the trick could be usefull in some cases
        #raw_data=raw_data[raw_data[:,0].argsort()]

        float_data=copy.deepcopy(raw_data)
        # replace nan values by previous float value
        for i in range(float_data.shape[0]):
            for j in range(1,self._nbf,1):
                if math.isnan(float_data[i,j]) :
                    float_data[i,j]=float_data[i-1,j]

        # last sanity check is to regularize the timestep as some steps can be missing
        # is the first point missing ?
        if self._uts < float_data[0,0]:
            index=(float_data[0,0]-self._uts)//self._step_in_s
            for k in range(index):
                for j in range(self._nbf):
                    self._full_data[k,j]=float_data[k,j]
        else:
            index=1
            for j in range(self._nbf):
                self._full_data[0,j]=float_data[0,j]

        for i in range(1,self._nbp,1):
            self._full_data[i,0]=self._full_data[i-1,0]+self._step_in_s
            # if full_data timestep is greater than or equal to the timestep of float_data at index i,
            # we can record a new value in full_data and increment index i
            if self._full_data[i,0] >= float_data[index,0] and index <= float_data.shape[0]-2:
                for j in range(1,self._nbf,1):
                    self._full_data[i,j]=float_data[index,j]
                index+=1
            else:
                for j in range(1,self._nbf,1):
                    self._full_data[i,j]=self._full_data[i-1,j]
                #print("delta is {} for i {}".format(float_data[index,0]-self._full_data[i,0],i))
        if view['vis']:
            xrange=np.arange(raw_data.shape[0])
            ax1=plt.subplot(311)
            plt.plot(raw_data[:,1],color="red")
            plt.scatter(xrange,raw_data[:,1],color="orange",marker='o',s=4)
            ax2=ax1.twinx()
            plt.plot(raw_data[:,0],color="blue",label="unixtimestamp")
            plt.legend(loc='upper right')
            plt.subplot(312,sharex=ax1)
            plt.plot(float_data[:,1],color="orange")
            ax3=plt.subplot(313)
            plt.plot(self._full_data[:,1],color="orange",label=view['lib'][0])
            plt.legend(loc='upper left')
            ax4=ax3.twinx()
            plt.plot(self._full_data[:,0],color="blue",label="unixtimestamp")
            plt.legend(loc='upper right')
            plt.show()

            plt.subplot(111)
            for j in range(len(view['lib'])):
                plt.plot(self._full_data[:,j+1],label=view['lib'][j])
            plt.legend(loc='upper left')
            plt.show()

    def save(self):
        """
        to be implemented
        """
        print("I want to save my datas")


"""
# example of use case
# retrieve nebulosity and temperature datas for the year 2018 and for the Clermont-Ferrand station
dataset='donnees-synop-essentielles-omm%40public'
# clermont-ferrand station is number 07460
# Lyon/Satolas(Colombier-Saugnieu) is number 07481 for example. It is the nearest station close to grenoble
station="07460"
start=2018
stop=2019
utz="Europe/Paris"
fields=["date","nbas","tc"]
# we fix here the presumed timestep in hour
# for data coming from météo france, timestep is usually 3 hours
step_in_h=3
source=openData(dataset,station,start,stop,fields,utz,step_in_h)
view={'vis':True,'lib':["nebulosity in Octa","external temperature in°C"]}
source.retrieve(view=view)
"""
