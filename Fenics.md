# Install Fenics on Ubuntu 18.04

## docker

```
sudo apt-get update
sudo apt-get remove docker docker-engine docker.io
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```
le système doit répondre :

```
alexandrecuer@alexandrecuer-PORTEGE-R30-A:~$ sudo systemctl enable docker
Created symlink /etc/systemd/system/multi-user.target.wants/docker.service → /lib/systemd/system/docker.service.

```

test de la version :
```
docker --version
```
le retour devrait être du type :
```
Docker version 19.03.6, build 369ce74a3c
```

pour obtenir la liste des containers:
```
sudo docker ps --all
```
quant il n'y en a pas encore de crée, le retour est le suivant :
```
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES 
```
## fenics container

```
curl -s https://get.fenicsproject.org | bash
```
le retour devrait être le suivant :
```
Successfully installed the fenicsproject script in /home/alexandrecuer/.local/bin/fenicsproject.

To get started, run the command

  fenicsproject run

For more information, see fenicsproject help.
```
