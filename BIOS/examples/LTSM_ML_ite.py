from src.modelstats import BuildingZone, GoToTensor
import numpy as np
import matplotlib.pylab as plt

dir="../phpfina"

"""
pavillon ITE
"""
nbptinh=1
step=3600//nbptinh

history_size=10
# we can use target_size to make a multi-size prediction
# when modifying target_size or history_size, the network has to be retrained !!!!
target_size=1
# how many hours in the future are we going to simulate ?
goto=200
#***********************************************************************************
#***********************************************************************************

params=[{"id":1,"action":"smp"},{"id":167,"action":"smp"},{"id":145,"action":"acc"}]

"""
zone 1 is composed of 60 days on october and november 2018, at the beginning of winter
zone 2 is composed of 51 days on december 2018 and january 2019
the tenants left the house for something like 300 hours at the beginning of january, leaving the temperature decrease
zone 2 is therefore more complete as far as the phenomena we want to study is concerned : we can say it contains a real diversity of datas
so we are going to train on zone2 and evaluate on zone1
"""
zone1=GoToTensor(params,step,1538640000,65*24*nbptinh,dir)
print(zone1.shape)
zone2=GoToTensor(params,step,1544184000,51*24*nbptinh,dir)
print(zone2.shape)

plt.subplot(211)
plt.title("zone1")
plt.plot(zone1[:,1])
plt.plot(zone1[:,0])
plt.subplot(212)
plt.title("zone2")
plt.plot(zone2[:,1])
plt.plot(zone2[:,0])
plt.show()

ite=BuildingZone(step,history_size,target_size)
ite.CalcMeanStd(zone2)
ite.AddSets(zone2, forTrain=True, shuffle=False)
ite.AddSets(zone1, forTrain=False)
ite.LSTMfit("ite_no_shuffle")

ite.MLAfit(zone2, regularize=False)
ite.MLAviewWeights()

MLA_datas, MLA_labels = ite.MLAprepare(zone1)
samples=[0,200,300,500,800,1200,1340]
for i in samples:
    LSTM_preds, LSTM_labels=ite.LSTMpredict(i,goto)
    MLA_preds=ite.MLApredict(MLA_datas,i,goto)
    ite.view(zone1, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds,
                                                MLAtruths=MLA_labels[i:i+goto], LSTMtruths=LSTM_labels)

zone3=GoToTensor(params,step,1538640000,150*24*nbptinh,dir)
ite.ClearSets(train=False)
ite.AddSets(zone3, forTrain=False)
MLA_datas, MLA_labels = ite.MLAprepare(zone3,labelsToPhysicsValue=True)
samples=[2400]
for i in samples:
    LSTM_preds, LSTM_labels=ite.LSTMpredict(i,goto)
    MLA_preds=ite.MLApredict(MLA_datas,i,goto)
    ite.view(zone3, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds,
                                                MLAtruths=MLA_labels[i:i+goto], LSTMtruths=LSTM_labels)
