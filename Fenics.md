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
alexandrecuer@alexandrecuer-PORTEGE-R30-A:~$ docker --version
Docker version 19.03.6, build 369ce74a3c
```
