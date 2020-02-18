# Allier Habitat dataset

collected over one year (June 2018 to august 2019)

3 single-family homes with electric heating
``` 
git clone https://github.com/alexandrecuer/tf_works
cd tf_works/BIOS
wget https://raw.githubusercontent.com/alexandrecuer/smartgrid/master/datasets/emoncms-backup-2019-08-19.tar.gz
tar -xvf emoncms-backup-2019-08-19.tar.gz
``` 
## finding a model

A model is needed before we can tackle the construction of the intelligent agent that will operate the HVAC. 

The agent will learn what to do using reinforcement learning techniques.

naive approach = next temperature in the room equals temperature at previous step
-  
2 options explored :

### multilinear regression
```
python3 mlreg.py
```
- 
- blackbox approach with supervised machine learning

```
python3 heating.py
```
