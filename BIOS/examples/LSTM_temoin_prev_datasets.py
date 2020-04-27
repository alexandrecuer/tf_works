from src.modelstats import BuildingZone, GoToTensor
import numpy as np
import matplotlib.pylab as plt

dir="../phpfina"

"""
This is a basic example to show how datasets for neural network are created
Allier Habitat dataset - pavillon temoin
"""
nbptinh=1
step=3600//nbptinh

history_size=10
# how many hours in the future are we going to simulate ?
goto=200

"""
here we inject the datasets in the training memory
the process is the same for validation datasets but when feeding the validation memory, do not recalculate the mean/std !! (skip step 3)
1) create the GoToTensor from the PHPFina feeds
2) initialize the object
3) calculate mean/std for regularization
4) build the datasets and add them to the memory

NOTA : as we are on the training memory, we have to fix shuffle=True !!!
if not set to True, all datasets will be in disorder and it will not be possible to compare with the data from the original GoToTensor
"""
paramsb=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
winter=GoToTensor(paramsb,step,1539950400,100*24*nbptinh,dir)
temoin=BuildingZone(step,history_size,1)
temoin.CalcMeanStd(winter)
temoin.AddSets(winter, forTrain=True, shuffle=False)

nbset=200
values=temoin._train_datas[nbset]*temoin._std+temoin._mean
"""
all this is supervized learning
convention : the dataset is in the past
the history range includes history_size points from -history_size to -1
the label is indoor true monitored temperature at t=0
"""
history_range=list(range(-temoin._history_size, 0))

ax1=plt.subplot(111)
ax1.set_ylabel('°C')

"""
plot the label of the dataset (indoor temp) at t=0 as a green point
"""
truefuture=temoin._train_labels[nbset]*temoin._std[1]+temoin._mean[1]
plt.plot(0,truefuture, 'o', label='true future', color=temoin._col[1])

"""
zero is the index of the end of the dataset in the GoToTensor
the dataset starting index is its number : nbset
"""
zero=nbset+temoin._history_size

"""
1) plot the dataset on history_range, as green, blue and red crosses

2) plot the datas coming from the GoToTensor as lines, with the same colors

- blue = outdoor temp (field 0)
- green = indoor temp (field 1)
- red = energy in kwh (field 2)

temperatures are to be considered as values at the step
energy consumption is to be considered as the value for the step to come

"""
for k in range(3):
    if k == 2:
       ax1.tick_params(axis='y')
       plt.legend()
       ax2 = ax1.twinx()
       ax2.set_ylabel("Kwh")
    plt.plot(history_range,values[:,k], 'rx', color=temoin._col[k])
    plt.plot(history_range,winter[nbset:zero,k], color=temoin._col[k], label=temoin._lab[k])
ax2.tick_params(axis='y')
plt.legend()
plt.show()

"""
all this is to illustrate how the view method of the BuildingZone class is working
"""
