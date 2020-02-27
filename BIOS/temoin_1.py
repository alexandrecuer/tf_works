from building import BuildingZone, GoToTensor
import numpy as np

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
# if singlesim is set to true, a single simulation is done and the goto param is not used
goto=48
singlesim=False

debug=False
training=True

# we cannot mix multi-simulation and muti-size prediction
if singlesim==False and target_size>1:
    print("cannot go further - in multisim mode, you need to fix target_size=1")
    sys.exit()
#***********************************************************************************
#***********************************************************************************

paramsb=[{"id":1,"action":"smp"},{"id":191,"action":"smp"},{"id":139,"action":"acc"}]
winter=GoToTensor(paramsb,step,1539950400,210*24*nbptinh)
tsize=winter.shape[0]-1-1500
temoin=BuildingZone(step,history_size,target_size)
temoin.CalcMeanStd(winter[0:tsize])

temoin.AddSets(winter[0:tsize+1],regularize=True, forTrain=True)
temoin.AddSets(winter[tsize-history_size:],regularize=True, forTrain=False)
temoin.LSTMfit("temoin")

#temoin.LSTMload("temoin")

temoin.ClearSets(train=False)
temoin.AddSets(winter,regularize=True, forTrain=False)
temoin.MLAfit(winter[0:tsize], regularize=True)
temoin.MLAviewWeights()

MLA_datas, MLA_labels = temoin.MLAprepare(winter,labelsToPhysicsValue=True)
samples=[200,1500,2000,3000,4220,4260,4300,4350,4400]
#for i in range(4260,winter.shape[0]-goto-1):
for i in samples:
    LSTM_preds, LSTM_labels=temoin.LSTMpredict(i,goto)
    MLA_preds=temoin.MLApredict(MLA_datas,i,goto)
    temoin.viewValSet(winter, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds,
                                                MLAtruths=MLA_labels[i:i+goto], LSTMtruths=LSTM_labels)
