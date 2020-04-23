# Install Fenics on Ubuntu 18.04

## docker

### installation

```diff
sudo apt-get update
sudo apt-get remove docker docker-engine docker.io
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
+Created symlink /etc/systemd/system/multi-user.target.wants/docker.service 
+→ /lib/systemd/system/docker.service.`
```

test de la version :
```diff
docker --version
+Docker version 19.03.6, build 369ce74a3c`
```

### ajouter le sudoer en cours dans le groupe docker

quant on lance docker sans sudo, on peut avoir le retour suivant :
```diff
+Got permission denied while trying to connect to the Docker daemon socket at 
+unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.40/volumes/create: 
+dial unix /var/run/docker.sock: connect: permission denied
```
il faut vérifier que l'user est bien dans le groupe docker
```
grep docker /etc/group
```

pour créer le groupe s'il n'existe pas (peu probable)
```
sudo groupadd docker
```

pour ajouter l'user en cours

```
sudo usermod -aG docker ${USER}
```

https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket

### hello world example

```
mkdir /home/alexandrecuer/fenics
cd /home/alexandrecuer/fenics
```

```diff
docker run hello-world
+Unable to find image 'hello-world:latest' locally
+latest: Pulling from library/hello-world
+0e03bdcc26d7: Pull complete 
+Digest: sha256:8e3114318a995a1ee497790535e7b88365222a21771ae7e53687ad76563e8e76
+Status: Downloaded newer image for hello-world:latest

+Hello from Docker!
+This message shows that your installation appears to be working correctly.

+To generate this message, Docker took the following steps:
+ 1. The Docker client contacted the Docker daemon.
+ 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
+    (amd64)
+ 3. The Docker daemon created a new container from that image which runs the
+    executable that produces the output you are currently reading.
+ 4. The Docker daemon streamed that output to the Docker client, which sent it
+    to your terminal.

+To try something more ambitious, you can run an Ubuntu container with:
+ $ docker run -it ubuntu bash

+Share images, automate workflows, and more with a free Docker ID:
+ https://hub.docker.com/

+For more examples and ideas, visit:
+ https://docs.docker.com/get-started/
```
pour obtenir la liste des containers:
```diff
sudo docker ps --all
+CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                   PORTS               NAMES
+a8f846d82c74        hello-world         "/hello"            2 hours ago         Exited (0) 2 hours ago                       practical_banzai
```


## fenics container

```diff
curl -s https://get.fenicsproject.org | bash
+Successfully installed the fenicsproject script in /home/alexandrecuer/.local/bin/fenicsproject.
+To get started, run the command
+  fenicsproject run
```




