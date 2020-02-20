# operating systems with machine intelligence

just a place to gather all IA works (mostly related to tensor flow but not only) - started in 2020

## Configuration

a "weak" machine

Toshiba Portégé R30A19J 64 bits :
- RAM : 7,7 Go
- HDD : 491,2 Go
- CPU : Intel Core i3-4100M CPU @ 2.50 Ghz x 4
- graphics : Intel Haswell Mobile

OS : Ubuntu 18.04 LTS

### python & tensorflow

```
sudo apt install python
sudo apt install python-pip
sudo apt install python3-pip
```

add to ~/.bashrc
```
alias python='/usr/bin/python3'
alias pip='/usr/bin/pip3'
```
install tensorflow and numpy
```
pip3 install tensorflow
pip3 install --upgrade tensorflow
pip3 install --upgrade numpy
```
with a CPU-only machine (**caution - seem to be deprecated**) :
```
pip3 install tensorflow-cpu
```

check keras version
```
pip3 list | grep -i keras
```

### install atom

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
pip install pandas
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
```
