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

paramsb=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
winter=GoToTensor(paramsb,step,1539950400,210*24*nbptinh,dir)
tsize=winter.shape[0]-1-1500
temoin=BuildingZone(step,history_size,target_size)
temoin.CalcMeanStd(winter[0:tsize])
# the following is optional as by default, datasets stored are regularized
#temoin.regularizeSets(True)

plt.subplot(211)
plt.title("winterstart")
plt.plot(winter[0:tsize:,1])
plt.plot(winter[0:tsize:,0])
plt.subplot(212)
plt.title("winterend")
plt.plot(winter[tsize-history_size:,1])
plt.plot(winter[tsize-history_size:,0])
plt.show()

temoin.AddSets(winter[0:tsize+1], forTrain=True)
temoin.AddSets(winter[tsize-history_size:], forTrain=False)
temoin.LSTMfit("temoin")

#temoin.LSTMload("temoin")

temoin.ClearSets(train=False)
temoin.AddSets(winter, forTrain=False)
temoin.MLAfit(winter[0:tsize], regularize=True)
temoin.MLAviewWeights()

MLA_datas, MLA_labels = temoin.MLAprepare(winter,labelsToPhysicsValue=True)
samples=[200,1500,2000,3000,4220,4260,4300,4350,4400]
for i in samples:
    LSTM_preds, LSTM_labels=temoin.LSTMpredict(i,goto)
    MLA_preds=temoin.MLApredict(MLA_datas,i,goto)
    temoin.view(winter, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds)
    """
    # use this if you want to check that labelling process is correct
    temoin.view(winter, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds,
                                                MLAtruths=MLA_labels[i:i+goto], LSTMtruths=LSTM_labels)
    """
