# tensor flow works

just a place to gather all work related to tensor flow - started in 2020

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






