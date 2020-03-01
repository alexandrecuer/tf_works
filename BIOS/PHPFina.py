from datetime import datetime
import math
import struct

# convert a unix time stamp expressed in seconds to something human readable
def humanDate(uts):
    human=datetime.utcfromtimestamp(uts).strftime('%Y-%m-%d %H:%M:%S')
    return human

# is this really needed or would we go for struct.unpack all the time ?
def readBytes(nb,file):
    bytes = file.read(4)
    value=int.from_bytes(bytes, byteorder='little', signed=False)
    return value

# for debugging in case
def checkPoints(nb,position,interval):
    with open("phpfina/"+str(nb)+".dat", "rb") as ts:
        for i in range(300):
            nbPtInHour=60*60/interval
            offset=int((position+i)*4)
            print(offset)
            ts.seek(offset, 0)
            hexa = ts.read(4)
            aa= bytearray(hexa)
            value=struct.unpack('<f', aa)[0]
            print(value)

"""
Seek can be called one of two ways:
x.seek(offset)
x.seek(offset, starting_point)

starting_point can be 0, 1, or 2
 0 - Default. Offset relative to beginning of file
 1 - Start from the current position in the file
 2 - Start from the end of a file (will require a negative offset)
"""
# import and manage emoncms PHPFINA objects
class PHPFina:
    # the constructor - you give the number of the timeserie (nb) and a step parameter, used for the sampling
    def __init__(self,nb,step,dir="phpfina"):
        """ initialize the metas """
        #starttime and interval expressed in seconds
        #starttime is a unixtimestamp
        self._startTime = 0
        self._interval = 0
        self._nb = nb
        # the unixtimestamp in seconds at which you decide to start sampling
        self._SamplingPos = 0
        # the period in second at which you sample from the timeseries
        self._step = step
        self._dir = dir
        self._datas = []
        self._debug = False

    # stores and returns the metas
    def getMetas(self):
        with open(self._dir+"/"+str(self._nb)+".meta", "rb") as metas:
            # Seek a specific position in the file and read N bytes
            metas.seek(8, 0)
            interval=readBytes(4,metas)
            metas.seek(12, 0)
            startTime=readBytes(4,metas)
            if self._debug:
                print("startTime : {} = {}s".format(humanDate(startTime),startTime))
                print("interval : {}s".format(interval))
            self._startTime=startTime
            self._interval=interval

    def setStart(self,unixTimeStart):
        self._SamplingPos=int((unixTimeStart-self._startTime)/self._interval)
        if self._debug:
            print("Sampling will start on record number {}".format(self._nb,self._SamplingPos))
            print("\n")

    # stores an array of values extracted from the timeserie
    # nbSteps datas are sampled from _SamplingPos at a period equal to _step
    # a PHPFina file is made of 4 bytes float values, with NAN when nothing was recorded from the sensor
    def getDatas(self,nbSteps):
        start = self._SamplingPos
        nbPtInStep=self._step//self._interval
        if self._step/self._interval - nbPtInStep == 0:
            offset=0
        elif self._step/self._interval - nbPtInStep == 0.5:
            print("we have an half step offset")
            offset=1
        else:
            print("offset - we cannot get the datas - check steps/intervals")
            return
        position=int(start*4)
        with open(self._dir+"/"+str(self._nb)+".dat", "rb") as ts:
            for i in range(nbSteps):
                ts.seek(position, 0)
                hexa = ts.read(4)
                aa= bytearray(hexa)
                if len(aa)==4:
                  value=struct.unpack('<f', aa)[0]
                else:
                  print("unpacking problem {} len is {} position is {}".format(i,len(aa),position))
                if math.isnan(value):
                    #print("timeseries {} - there is a NAN at position {} in the timeserie or point number {}".format(self._nb,position,i))
                    j=1
                    while True:
                        ramble=position+j*4
                        ts.seek(ramble, 0)
                        hexa = ts.read(4)
                        aa= bytearray(hexa)
                        value=struct.unpack('<f', aa)[0]
                        if math.isnan(value):
                            j+=1
                        else:
                            break
                    #print("we found a value {} {}s after the NAN".format(value,self._interval*j))
                #print(position)
                #print(value)
                #input("press key")
                self._datas.append(value)
                if (i % 2) == 0:
                    position+=nbPtInStep*4
                else:
                    position+=(nbPtInStep+offset)*4

    # only for "energy" timeseries
    # accumulates the power for each step and outputs the energy consumption in Kwh within the step to come
    def getKwh(self,nbSteps):
        start = self._SamplingPos
        nbPtInStep=self._step//self._interval
        #print("an hour is sectionned in {} parts".format(nbPtInStep))
        with open(self._dir+"/"+str(self._nb)+".dat", "rb") as ts:
            for i in range(nbSteps+1):
                position=int((start+i*nbPtInStep)*4)
                acc=0
                ts.seek(position, 0)
                hexa = ts.read(4*nbPtInStep)
                aa= bytearray(hexa)
                if len(aa)==4*nbPtInStep:
                    values=struct.unpack('<{}f'.format(nbPtInStep), aa)
                for j in range(len(values)):
                    if math.isnan(values[j]):
                        val=0
                    else:
                        val=values[j]
                    acc+=val*self._interval
                if math.isnan(acc):
                    acc=0
                    print("BUG : consumption for full step is NAN !")
                # kwh conversion !!
                self._datas.append(0.001*acc/3600)

    # EmonCMS can provide the accumulated kwh feed over the whole recording period, like an energy meter.
    # from the accumulated kwh feed, the kwh consumed per hour can be recalculated
    # NOTA NOTA NOTA : recalculating from the instantaneous power is a better option
    # Indeed, there may be missing data in feeds dedicated to the accumulation of Kwh.
    # Experience shows that only instant power feeds ARE MANUALLY corrected in real time during monitoring.
    def unAcc(self):
        unAccvalues=[]
        for i in range(len(self._datas)-1):
            if self._datas[i+1]-self._datas[i] <0:
                print("bug")
                print(i)
                print("values are {} and {}".format(self._datas[i+1],self._datas[i]))
            unAccvalues.append(self._datas[i+1]-self._datas[i])
        self._datas=unAccvalues
