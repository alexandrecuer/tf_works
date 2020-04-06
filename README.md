# operating systems with machine intelligence

numerical analysis work related to the **BIOS** project - started in 2020

BIOS : Building Intelligent Operating System

## Main configuration

a "weak" machine

Toshiba Portégé R30A19J 64 bits :
- RAM : 7,7 Go
- HDD : 491,2 Go
- CPU : Intel Core i3-4100M CPU @ 2.50 Ghz x 4
- graphics : Intel Haswell Mobile

OS : Ubuntu 18.04 LTS

Remember to run the update commands with the ubuntu packetmanager or directly  
```
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get dist-upgrade -y
sudo apt-get clean
sudo apt --fix-broken install
```
## DataScience tools

### configuration with python 3.6

The Ubuntu 18.04 LTS ships with python3.6. you just have to install pip3
```
sudo apt install python3-pip
```
install scipy/matplotlib/numpy = matlab like tools for python

https://www.scipy.org/install.html

[SciPy](https://www.scipy.org/) is a Python-based ecosystem of open-source software for mathematics, science, and engineering

```
pip3 install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

install tensorflow
```
pip3 install tensorflow
pip3 install --upgrade tensorflow
```
with a CPU-only machine (**caution - seem to be deprecated**) :
```
pip3 install tensorflow-cpu
```

check keras version
```
pip3 list | grep -i keras
```
### upgrading to python 3.7


### Atom

```
wget -qO - https://packagecloud.io/AtomEditor/atom/gpgkey | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] https://packagecloud.io/AtomEditor/atom/any/ any main" > /etc/apt/sources.list.d/atom.list'
sudo apt-get update
sudo apt-get install atom
```

### openGym

```
pip install gym
```

### statistic tools

```
pip install -U scikit-learn
pip install statsmodels
```

## install virtualbox on Ubuntu Bionic

```
sudo nano /etc/apt/sources.list
```

add the following line at the end of sources.list :
```
deb [arch=amd64] https://download.virtualbox.org/virtualbox/debian bionic contrib
```

add the public key :
```
wget -q https://www.virtualbox.org/download/oracle_vbox_2016.asc -O- | sudo apt-key add -
```

install via apt :
```
sudo apt-get update
sudo apt-get install virtualbox-6.1
```
Launch virtualbox
```
virtualbox &
```

## Jupiter notebook
```
pip3 install jupyter
sudo apt-get install texlive-full
sudo apt-get install pandoc
```
Install some basic latex templates :
```
cd ~/.jupyter/
wget https://raw.githubusercontent.com/alexandrecuer/tf_works/master/jupyter_notebook_config.py
mkdir nbconvert_templates
cd nbconvert_templates
wget https://raw.githubusercontent.com/alexandrecuer/tf_works/master/revtex.tplx
```
This permits to add the following json section to your metadatas notebook
```
"latex_metadata": {
    "affiliation": "Dromotherm@Cerema",
    "author": "Alexandre CUER",
    "title": "indoor temperature prediction - using real datas to fit a model"
  }
```

launch the notebook, from the folder where you want to store your notebooks :
```
jupyter notebook
```
open mozilla and enter the following address :
```
http://localhost:8888
```
### latex templates

for more information  :

https://michaelgoerz.net/notes/custom-template-for-converting-jupyter-notebooks-to-latex.html

http://blog.juliusschulz.de/blog/ultimate-ipython-notebook#converting-notebook-to-latex-pdf

https://www.xm1math.net/doculatex/espacements.html
