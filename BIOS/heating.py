import numpy as np
import random
import math
import time
import struct
import matplotlib.pylab as plt
from PHPFina import PHPFina,humanDate

winterStart=1539950400
#winterStart=1546974000
print ("début hiver : {} ou {}s en unixtimestamp".format(humanDate(winterStart),winterStart))

def InitializeFeed(nb,step,start):
    feed=PHPFina(nb,step)
    feed.getMetas()
    feed.setStart(start)
    return feed

# given some PHPFina feeds, a period and a start (as a unix timestamp) both in seconds
def GoToTensor(params,step,start,nbsteps):
    print("going to tensor for {} feeds".format(len(params)))
    float_data=np.zeros((nbsteps,len(params)))
    for i in range(len(params)):
        print("feed number {}".format(i))
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

def ChecknR(tensor,regularize=True):
    mean=tensor.mean(axis=0)
    if regularize:
        tensor-=mean
    std=tensor.std(axis=0)
    if regularize:
        tensor/=std
    print(mean)
    print(std)

# outdoor, indoor, instant power
params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
## Sampling feeds and tensor construction
# we assume the winter period to be something like 200 or 210 days
# FIX THE NUMBER OF POINTS PER HOUR !
nbptinh=1
nbsteps=300*24*nbptinh
step=3600//nbptinh
winter=GoToTensor(params,step,winterStart,nbsteps)

plt.subplot(311)
#Out temperature
plt.plot(winter[:,0])
#In temperature
plt.plot(winter[:,1])
plt.ylabel("In and Out Temp in °C")
plt.subplot(312)
plt.ylabel("Kwh used for heating per step")
plt.plot(winter[:,2])
plt.xlabel("Steps - one step = {}s".format(step))
plt.subplot(313)
plt.scatter(winter[:,0],winter[:,2], color='red')
plt.show()

# first approach - Multiple Linear Regression
from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm
l=winter.shape[0]-1
futureTint=[]
for i in range(l):
    futureTint.append(winter[i+1,1])

# regression with sklearn
regr = linear_model.LinearRegression()
regr.fit(winter[:l,:], futureTint)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# regression with statsmodels
b = sm.add_constant(winter[:l,:]) # adding a constant
model = sm.OLS(futureTint, b).fit()
predictions = model.predict(b)
print_model = model.summary()
print(print_model)

input("press key")

# ***********
# a generator
def heating_gen(data,rows,lookback,delay,batch_size=50):
    i=0
    while 1:
        if i+batch_size>len(rows):
            i=0
        i+=batch_size
        samples=np.zeros((batch_size,lookback,data.shape[-1]))
        targets=np.zeros((batch_size,))
        for j in range(batch_size):
            indices = range(rows[j]-lookback, rows[j])
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay-1][1]
        yield samples, targets

# the same but with everything in RAM
def heating_simple(data,rows,lookback,delay):
    samples=np.zeros((len(rows),lookback,data.shape[-1]))
    targets=np.zeros((len(rows),))
    for j in range(len(rows)):
        indices = range(rows[j]-lookback, rows[j])
        samples[j]=data[indices]
        targets[j]=data[rows[j]+delay-1][1]
    return samples, targets

def plot_batch(samples,targets,i):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time')
    ax1.set_ylabel('T')
    ax1.plot(samples[i,:,0], color='tab:red')
    ax1.plot(samples[i,:,1], color='tab:blue')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Kwh/h')
    ax2.plot(samples[i,:,2], color='tab:green')
    ax2.tick_params(axis='y')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    target="%.2f" %targets[i]
    plt.suptitle("el. {} with target {}".format(i,target))
    plt.show()

def plot(samples,target,text):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time')
    ax1.set_ylabel('T')
    ax1.plot(samples[:,0], color='tab:red')
    ax1.plot(samples[:,1], color='tab:blue')
    for j, v in enumerate(samples[:,1]):
        ax1.text(j, v, "%.1f" %v, bbox=dict(facecolor='blue', alpha=0.1))
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Kwh/h')
    ax2.plot(samples[:,2], color='tab:green')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    plt.suptitle("{} - target {}".format(text,"%.2f" %target))
    plt.show()

# we create the sampling arrays for training and validation
start=20*nbptinh
size=24*3*7*nbptinh
period=24*4*7*nbptinh
# fix the jump to go from one dataset to another
jump=nbptinh
for i in range(10):
    if i==0:
        t=np.arange(start,start+size,jump)
        v=np.arange(start+size,start+period,jump)
    else:
        t=np.concatenate((t,np.arange(start,start+size,jump)),axis=None)
        v=np.concatenate((v,np.arange(start+size,start+period,jump)),axis=None)
    start+=period
nts=len(t)
nvs=len(v)
print(t)
print("number of training samples : {}".format(nts))
print(v)
print("number of validation samples : {}".format(nvs))
ChecknR(winter,False)
batch_size=50
lookback=12
#heating_train=heating_gen(winter,t,lookback*nbptinh,nbptinh,batch_size)
#heating_val=heating_gen(winter,v,lookback*nbptinh,nbptinh,batch_size)
#t_steps=nts//batch_size
#v_steps=nvs//batch_size
x_train,y_train=heating_simple(winter,t,lookback*nbptinh,nbptinh)
x_val,y_val=heating_simple(winter,v,lookback*nbptinh,nbptinh)


for i in range(10):
    plot(x_train[i],y_train[i],"training el. {}".format(i))

for i in range(10):
    plot(x_val[i],y_val[i],"validation el. {}".format(i))

input("press a key")

import tensorflow as tf
print("TF", tf.__version__)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(lookback*nbptinh,winter.shape[-1])))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
#model.add(tf.keras.layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,winter.shape[-1])))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss='mae')

class History(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

history = History()

start_time = time.time()
model.fit(x_train,y_train,batch_size=50,epochs=40,validation_data=(x_val,y_val),callbacks=[history])
model.evaluate(x_val, y_val, verbose=1)

print("Execution time in %s seconds ---" % (time.time() - start_time))

loss=history.losses
val_loss=history.val_losses
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs,loss,'bo',label='training loss')
plt.plot(epochs,val_loss,'b',label='validation loss')
plt.title("validation and training loss")
plt.legend()
plt.show()
