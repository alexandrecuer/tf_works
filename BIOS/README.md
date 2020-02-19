# Allier Habitat dataset

collected over one year (June 2018 to august 2019)

3 single-family homes with electric heating
``` 
git clone https://github.com/alexandrecuer/tf_works
cd tf_works/BIOS
wget https://raw.githubusercontent.com/alexandrecuer/smartgrid/master/datasets/emoncms-backup-2019-08-19.tar.gz
tar -xvf emoncms-backup-2019-08-19.tar.gz
```
the problem : Regarding comfort and energy savings, would it be possible to improve the operation of HVAC systems, in the field of **tertiary buildings**, using reinforcement learning (RL) techniques ? Could it be possible to train an intelligent agent in a sandbox running an appropriate model and then to drop it on the field ?

A model is needed before we can tackle the construction of an intelligent agent that will operate the HVAC. 
We have to keep in mind the difficulty to acquire representative datas for each phenomena we need to describe in order to properly cover our problematic (i.e. the energy supply of the building). A range of 1 year of real-life datas is a small thing as real-life is very different from gaming experience, a field where RL has proven to be efficient. So we need to compensate using modelization....

An hour is a reasonable timestep, so :
- t = 1 means t + 1 hour
- t = plus48 means t + 48 hours (a full weekend range)

The agent will have to estimate PKwh(t in 1 -> plus48) in order to reach, at t = plus48, the confort temperature setpoint, given some history (monitoring) and simulation inputs...

monitored data inputs :
- PKwh(t in -10 -> 0) 
- Text(t in -10 -> 0)
- Tint(t in -10 -> 0)

simulated data inputs :
- Text(t in 1 -> plus48)

in offline reinforcement learning mode, a typical step can be described as follow :
- the agent will calculate PKwh(t in 1 -> plus48)
- the model will evaluate Tint(t in 1 -> plus48)
- the system will rate and record the performance in a memory
- the system will use the memory to create batches on the fly
- the system train the agent on a random batch

On the field, the monitored thruth (Tint) will replace the model

## finding a model

For a given building, the model has to find the "next" correct internal temperature, ie Tint(t =1), given some history :
- Tint(t in [minus24 -> 0])
- Text(t in [minus24 -> 0])
- PKwh(t in [minus24 -> 0])
- Text(t = 1)

### naive approach

next temperature in the room equals temperature at previous step - OK for t = 1, not more

### multilinear regression
```
python3 mlreg.py
```

### blackbox approach with supervised machine learning

```
python3 heating.py
```
