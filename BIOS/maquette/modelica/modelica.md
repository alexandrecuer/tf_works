# openmodelica

https://openmodelica.org

`OMEdit &`

Outils > options > Terminal Command : `/usr/bin/OMShell-terminal`

Les modèles modelica sont des fichiers texte avec une extension en .mo

on peut simuler un modèle en ligne de commande...

dans un dossier contenant un fichier test.mo, créer un fichier script.mos avec les instructions suivantes :

```
loadModel(Modelica); 
loadFile("test.mo");
simulate(test);
```
si on veut plus de détails, on peut rajouter des options à la commande simulate :
```
simulate(test, simflags="-d=aliasConflicts");
```

Pour spécifier la durée de simulation et le nombre d'intervalles :
```
simulate(test, startTime = 0, stopTime = 500, numberOfIntervals=500)
```
startTime et stopTime sont des valeurs en secondes

Toujours depuis le même dossier, lancer une console et lancer la commande suivante :

- sous linux : `omc script.mos`

- sous windows : `"%OPENMODELICAHOME%\bin\omc.exe" script.mos`

# exemples avec la bibliothèque standard

https://build.openmodelica.org/Documentation/Modelica.Fluid.Examples.PumpingSystem.html



# bibliothèques

généralement, ces bibliothèques embarquent la bibliothèque dite ibpsa https://github.com/ibpsa/modelica-ibpsa, maintenue par le lbl

IBPSA = International Building Performance Simulation Association


## bibliothèque du lbl (berkeley lab)

git clone https://github.com/lbl-srg/modelica-buildings

cette bibliothèque contient des vannes 3 voies

un seul exemple mais qui ne veut pas se compiler.... 

https://simulationresearch.lbl.gov/modelica/releases/v2.1.0/help/Buildings_Fluid_Actuators_Valves_Examples.html#Buildings.Fluid.Actuators.Valves.Examples.ThreeWayValves

## open-ideas

pas vraiment testé encore

git clone https://github.com/open-ideas/FastBuildings

git clone https://github.com/open-ideas/IDEAS

## EDF

git clone https://github.com/EDF-TREE/BuildSysPro.git

# couplage avec python

https://pdfs.semanticscholar.org/f6be/32042ee837eaad6cc7955854918c7ff6de5a.pdf

https://www.researchgate.net/publication/259902293_An_OpenModelica_Python_Interface_and_its_use_in_PySimulator

https://www.researchgate.net/publication/261259265
