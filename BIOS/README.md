Status : work in progress

the problem : Regarding comfort and energy savings, would it be possible to improve the operation of HVAC systems, in the field of **tertiary buildings**, using reinforcement learning (RL) techniques ? Could it be possible to train an intelligent agent in a sandbox running an appropriate model and then to drop it on the field ?

A model is needed before we can tackle the construction of an intelligent agent that will operate the HVAC. 
We have to keep in mind the difficulty to acquire representative datas for each phenomena we need to accurately describe in order to properly cover our problematic (i.e. the energy supply of the building). A range of 1 year of real-life datas is a small thing as real-life is very different from gaming experience, a field where RL has proven to be efficient. So we need to compensate using modelization....

# installation and datasets addition

Some datasets are available here : https://alexandrecuer.github.io/smartgrid/

To install them :

``` 
git clone https://github.com/alexandrecuer/tf_works
cd tf_works/BIOS
wget https://raw.githubusercontent.com/alexandrecuer/smartgrid/master/datasets/emoncms-backup-2019-08-19.tar.gz
tar -zxvf emoncms-backup-2019-08-19.tar.gz phpfina
mkdir labo
cd labo
wget https://raw.githubusercontent.com/alexandrecuer/smartgrid/master/datasets/emoncms-backup-2020-04-22.tar.gz
tar -zxvf emoncms-backup-2020-04-22.tar.gz phpfina
```

to install the src module

```
cd ..
```
then launch `pip install -e ./` or `python3 setup.py develop`

This will create a link named `src.egg-link` :
- in `/usr/local/lib/python3.x/dist-packages` on Linux
- in `C:\Users\...username....\AppData\Local\Programs\Python\Pythonxxx\Lib\site-packages` on Windows

**Please note :**
`python setup.py install` or `pip3 install .` are used to install (typically third party) packages that you're not going to develop/modify/debug yourself.

To check which modules are installed :
```
python
help("modules")
```

documentation for module src is available at : http://alexandrecuer.github.io/tf_works

generated with pdoc

```
pip3 install pdoc3
pdoc --html src
```
Once datasets and module are installed, you can play with the examples

# finding a model

For a given building, the model has to find the "next" correct internal temperature, ie Tint(t =1), given some history :
- Tint(t in [minus24 -> 0])
- Text(t in [minus24 -> 0])
- PKwh(t in [minus24 -> 0])

maybe also adding Text(t = 1) ???

### naive approach

Common sense tells us that indoor temperatures vary little compared to outdoor temperatures. 
So the naive approach could be to consider that next temperature in the room equals temperature at previous step. 
This approach cannot be implemented as we need to predict over one week or more

### blackbox approach using multilinear regression VS supervised machine learning

Two classes have been implemented : PHPFina and BuildingZone, plus one external method named GoToTensor
- PHPFina is used to sample the PHPFina hexa timeseries
- GoToTensor permits to create a numpy array/tensor from the samples 
- BuildingZone creates+stores the datasets and implements fitting and prediction using a basic multinear approach and/or a LSTM (Long Short term Memory) neural network 

Structure of a tensor is :
- tensor[:,0] = outdoor temperature values (at the step)
- tensor[:,1] = indoor temperature values (at the step)
- tensor[:,2] = energy consumptions (for the step to come) in kwh

the datasets are constructed by slicing the tensor at a specified index, on a specified length

How to use :

```
from src.modelstats import BuildingZone, GoToTensor
import numpy as np
```
define some sizing parameters
```
# number of points per hour
nbptinh=1
# timestep in seconds
step=3600//nbptinh
# history size in step(s) for prediction
history_size=10
# size of prediction in step - keep 1 !!!
target_size=1
# how many steps in the future are we going to simulate ?
goto=200
```
create some tensors from the PHPFina feeds

```
params=[{"id":1,"action":"smp"},{"id":167,"action":"smp"},{"id":145,"action":"acc"}]
test=GoToTensor(params,step,1538640000,65*24*nbptinh)
train=GoToTensor(params,step,1544184000,51*24*nbptinh)
```
create the datasets from the tensors and fit 

```
ite=BuildingZone(step,history_size,target_size)
ite.CalcMeanStd(train)
ite.AddSets(train, forTrain=True, shuffle=False)
ite.AddSets(test, forTrain=False)
ite.LSTMfit("ite_no_shuffle")
ite.MLAfit(train, regularize=False)
ite.MLAviewWeights()
```
check performance on test datas
```
MLA_datas, MLA_labels = ite.MLAprepare(test)
samples=[0,200,300,500,800,1200,1340]
for i in samples:
    LSTM_preds, LSTM_labels=ite.LSTMpredict(i,goto)
    MLA_preds=ite.MLApredict(MLA_datas,i,goto)
    ite.view(test, i, goto, MLApreds=MLA_preds, LSTMpreds=LSTM_preds,
                                                MLAtruths=MLA_labels[i:i+goto], LSTMtruths=LSTM_labels)
```
