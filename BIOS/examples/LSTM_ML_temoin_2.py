from src.modelstats import BuildingZone, GoToTensor
import numpy as np
import matplotlib.pylab as plt

dir="../phpfina"

"""
pavillon temoin
"""
nbptinh=1
step=3600//nbptinh

history_size=10
# we can use target_size to make a multi-size prediction
# when modifying target_size or history_size, the network has to be retrained !!!!
target_size=1
# how many hours in the future are we going to simulate ?
goto=200

params=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
zone1=GoToTensor(params,step,1537876800,148*24*nbptinh,dir)
zone2=GoToTensor(params,step,1550664000,83*24*nbptinh,dir)
temoin=BuildingZone(step,history_size,target_size)

plt.subplot(211)
plt.title("winterstart")
plt.plot(zone1[:,1])
plt.plot(zone1[:,0])
plt.subplot(212)
plt.title("winterend")
plt.plot(zone2[:,1])
plt.plot(zone2[:,0])
plt.show()

temoin.CalcMeanStd(zone2)
# the following is optional as by default, datasets stored are regularized
#temoin.regularizeSets(True)
temoin.AddSets(zone1, forTrain=True, shuffle=False)
temoin.AddSets(zone2, forTrain=False)
temoin.LSTMfit("temoin_2")
#temoin.LSTMload("temoin_2")

temoin.MLAfit(zone1, regularize=False)
temoin.MLAviewWeights()

MLA_datas, MLA_labels = temoin.MLAprepare(zone2)
samples=[0,200,500,1000,1500,1750]
for i in samples:
    LSTM_preds, LSTM_labels=temoin.LSTMpredict(i,goto)
    MLA_preds=temoin.MLApredict(MLA_datas,i,goto)
    temoin.view(zone2, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds,
                                                MLAtruths=MLA_labels[i:i+goto], LSTMtruths=LSTM_labels)
