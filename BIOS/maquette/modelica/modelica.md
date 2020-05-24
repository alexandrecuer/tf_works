# Links

http://www.lirmm.fr/~reitz/Modelica/

https://mbe.modelica.university/

https://sti.discip.ac-caen.fr/spip.php?article250

https://www.maplesoft.com/documentation_center/online_manuals/modelica/Modelica_Thermal_HeatTransfer_Components.html

https://www.abcclim.net/vanne-23-voies.html

http://www.valvias.com/flow-equations-flow-coefficient-cv-kv.php

http://www.waterblast.com/uploadedFiles/Site/Service_and_Support/Resources/40KCat-SecH-Technical.pdf

# Openmodelica

https://openmodelica.org

`OMEdit &`

Outils > options > Terminal Command : `/usr/bin/OMShell-terminal`

# Langage modelica et modèles

## Edition des fichier .mo

Les modèles modelica sont des fichiers texte avec une extension en .mo

On peut les ouvrir avec un éditeur de texte comme [notepad++](https://notepad-plus-plus.org) ou [Atom](https://atom.io/)

## Simulation en ligne de commande

Vu l'ergonomie de l'interface graphique, on peut choisir de simuler un modèle en ligne de commande

Avant de construire un modèle, on crée un répertoire qui lui est propre....

par exemple, créer un dossier `ModModelica` dans son répertoire de travail, puis à l'intérieur un sous-dossier `script1`

Avec openmodelica, on construit un modèle `test.mo` et on le sauve dans le dossier `script1`

Avec un éditeur de texte, toujours dans le dossier `script1`, créer un fichier script.mos avec les instructions suivantes :

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

Toujours depuis le même dossier, ouvrir une console et lancer la commande suivante :

- sous linux : `omc script.mos`

- sous windows : `"%OPENMODELICAHOME%\bin\omc.exe" script.mos`

Celà va créer beaucoup de fichiers C....

Pour consulter les résultats, il suffit d'ouvrir le fichier de résultat (extension .mat) avec openmodelica....

on peut afficher les courbes dans l'onglet tracé

##  Exemples de fichier modèle

Un exemple avec des fluides est disponible [içi](script1/cuves_gravitaires.mo)
- faire une simulation sur 500 secondes et afficher le volume des cuves ainsi que l'évolution des pressions

Un autre plutôt thermique avec un [modèle très rudimentaire de bâtiments](script2/test_house.mo)
- faire une simulation sur 1e7 secondes et afficher température intérieure, température du mur et température extérieure

Quant on travaille avec des fluides, il est important de spécifier le type de fluide et que l'éditeur graphique ne le fait pas. Pour celà, au tout début de la classe, on définit le fluide :
```
model cuves_gravitaires
  replaceable package Water = Modelica.Media.Water.ConstantPropertyLiquidWater;
```
A chaque intégration d'un nouvel élément dans le modèle, si celui ci véhicule du fluide, il faut le spécifier et faire référence au fluide défini au départ.

Avec un élément de type OpenTank :

```
Modelica.Fluid.Vessels.OpenTank CuvePleine(
    redeclare package Medium = Water,
```

A noter que les cuves sont dotés d'un système de ports et que la gestion des ports est l'affaire de l'utilisateur. Le bloc de code ci-dessous permet la définition d'une cuve de 2 mètres de haut, de 1 m2 de diamètre, pleine à moitié (1 m) et dotée de 3 ports :
- deux de 10 centimètres de diamètre, pour les tuyaux
- le troisième de 1 cm de diamètre, pour un capteur de pression relative

```
Modelica.Fluid.Vessels.OpenTank CuvePleine(
    redeclare package Medium = Water,
    crossArea = 1,
    height = 2,  
    level_start = 1,
    nPorts = 3,
    portsData = {Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.1),
                   Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.1),
                   Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.01)}) annotation(
    Placement(visible = true, transformation(origin = {-50, 34}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
```
Le modèle se termine toujours par un instruction end :
```
end cuves_gravitaires;
```

# Exemples issus de la bibliothèque standard

https://github.com/modelica/ModelicaStandardLibrary

https://build.openmodelica.org/Documentation/Modelica.Fluid.Examples.PumpingSystem.html



# Bibliothèques spécifiques bâtiments

généralement, ces bibliothèques embarquent la bibliothèque dite ibpsa https://github.com/ibpsa/modelica-ibpsa, maintenue par le lbl

IBPSA = International Building Performance Simulation Association

## Bibliothèque du lbl (berkeley lab)
```
git clone https://github.com/lbl-srg/modelica-buildings
```
cette bibliothèque contient des vannes 3 voies

un seul exemple mais qui ne veut pas se compiler.... 

https://simulationresearch.lbl.gov/modelica/releases/v2.1.0/help/Buildings_Fluid_Actuators_Valves_Examples.html#Buildings.Fluid.Actuators.Valves.Examples.ThreeWayValves

## Open-ideas

pas vraiment testé encore
```
git clone https://github.com/open-ideas/FastBuildings

git clone https://github.com/open-ideas/IDEAS
```
## EDF
```
git clone https://github.com/EDF-TREE/BuildSysPro.git
```
# Couplage avec python

https://pdfs.semanticscholar.org/f6be/32042ee837eaad6cc7955854918c7ff6de5a.pdf

https://www.researchgate.net/publication/259902293_An_OpenModelica_Python_Interface_and_its_use_in_PySimulator

https://www.researchgate.net/publication/261259265

https://github.com/OpenModelica/OMPython
