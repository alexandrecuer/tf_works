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

relancer la machine

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

On télécharge le script d'installation
```diff
curl -s https://get.fenicsproject.org | bash
+Successfully installed the fenicsproject script in /home/alexandrecuer/.local/bin/fenicsproject.
+To get started, run the command
+  fenicsproject run
```

On télécharge l'image en lançant la commande `fenicsproject run` ce qui peut être assez long, vu qu'elle est volumineuse....

A noter que cette commande crée en suivant un container et le lance....

On se retrouvera donc à la fin de l'installation dans un environnement différent

On constate que le nom de l'image est quay.io/fenicsproject/stable:current
```diff
fenicsproject run
+WARNING: No swap limit support
+WARNING: No swap limit support
+[docker volume create --name instant-cache-quay.iofenicsprojectstablecurrent]

+instant-cache-quay.iofenicsprojectstablecurrent
+[docker create -ti -p 127.0.0.1:3000-4000:8000 --env HOST_UID=1000 --env HOST_GID=1000 --env MAKEFLAGS=-j1 -v instant-cache-+quay.iofenicsprojectstablecurrent:/home/fenics/.cache/fenics --env INSTANT_CACHE_DIR=/home/fenics/.cache/fenics/instant --env DIJITSO_CACHE_DIR=/home/fenics/.cache/fenics/dijitso -v '/home/alexandrecuer/fenics':/home/fenics/shared -w /home/fenics/shared --label org.fenicsproject.created_by_script=true quay.io/fenicsproject/stable:current '/bin/bash -l -c "export TERM=xterm; bash -i"']

+Unable to find image 'quay.io/fenicsproject/stable:current' locally
+current: Pulling from fenicsproject/stable
+c64513b74145: Pulling fs layer
+01b8b12bad90: Pulling fs layer
+c5d85cf7a05f: Pulling fs layer
+b6b268720157: Pulling fs layer
+e12192999ff1: Pulling fs layer
+d39ece66b667: Pulling fs layer
+65599be66378: Pulling fs layer
+04de8bc2d500: Pulling fs layer
+abb684b96e3d: Pulling fs layer
+5fb302170f66: Pulling fs layer
+56d9f5e23832: Pulling fs layer
+2362411179ee: Pulling fs layer
+f0f1cc16c840: Pulling fs layer
+c5bca2c84e5e: Pulling fs layer
+6f80705d1d37: Pulling fs layer
+91e158461d4c: Pulling fs layer
+4b299c049e4f: Pulling fs layer
+4642a6d46aeb: Pulling fs layer
+0526f57f6bb3: Pulling fs layer
+6bda00aab163: Pulling fs layer
+599370ebdfdb: Pulling fs layer
+3debe21df7b8: Pulling fs layer
+bd6da2b7ad00: Pulling fs layer
+3111ae43df8e: Pulling fs layer
+dd2003bd419e: Pulling fs layer
+49e25fdb34c3: Pulling fs layer
+d95fd6a8326e: Pulling fs layer
+65599be66378: Waiting
+04de8bc2d500: Waiting
+abb684b96e3d: Waiting
+5fb302170f66: Waiting
+56d9f5e23832: Waiting
+2362411179ee: Waiting
+f0f1cc16c840: Waiting
+c5bca2c84e5e: Waiting
+3debe21df7b8: Waiting
+6f80705d1d37: Waiting
+bd6da2b7ad00: Waiting
+91e158461d4c: Waiting
+3111ae43df8e: Waiting
+4b299c049e4f: Waiting
+dd2003bd419e: Waiting
+4642a6d46aeb: Waiting
+49e25fdb34c3: Waiting
+0526f57f6bb3: Waiting
+d95fd6a8326e: Waiting
+6bda00aab163: Waiting
+599370ebdfdb: Waiting
+b6b268720157: Waiting
+e12192999ff1: Waiting
+d39ece66b667: Waiting
+c5d85cf7a05f: Verifying Checksum
+c5d85cf7a05f: Download complete
+01b8b12bad90: Verifying Checksum
+01b8b12bad90: Download complete
+e12192999ff1: Verifying Checksum
+e12192999ff1: Download complete
+b6b268720157: Verifying Checksum
+b6b268720157: Download complete
+d39ece66b667: Verifying Checksum
+d39ece66b667: Download complete
+04de8bc2d500: Verifying Checksum
+04de8bc2d500: Download complete
+abb684b96e3d: Verifying Checksum
+abb684b96e3d: Download complete
+5fb302170f66: Verifying Checksum
+5fb302170f66: Download complete
+56d9f5e23832: Verifying Checksum
+56d9f5e23832: Download complete
+c64513b74145: Verifying Checksum
+c64513b74145: Download complete
+2362411179ee: Verifying Checksum
+2362411179ee: Download complete
+c64513b74145: Pull complete
+01b8b12bad90: Pull complete
+c5d85cf7a05f: Pull complete
+b6b268720157: Pull complete
+e12192999ff1: Pull complete
+d39ece66b667: Pull complete
+65599be66378: Verifying Checksum
+65599be66378: Download complete
+65599be66378: Pull complete
+04de8bc2d500: Pull complete
+abb684b96e3d: Pull complete
+5fb302170f66: Pull complete
+56d9f5e23832: Pull complete
+2362411179ee: Pull complete
+6f80705d1d37: Verifying Checksum
+6f80705d1d37: Download complete
+f0f1cc16c840: Retrying in 5 seconds
+91e158461d4c: Retrying in 5 seconds
+f0f1cc16c840: Retrying in 4 seconds
+91e158461d4c: Retrying in 4 seconds
+f0f1cc16c840: Retrying in 3 seconds
+91e158461d4c: Retrying in 3 seconds
+f0f1cc16c840: Retrying in 2 seconds
+91e158461d4c: Retrying in 2 seconds
+f0f1cc16c840: Retrying in 1 second
+91e158461d4c: Retrying in 1 second
+c5bca2c84e5e: Verifying Checksum
+c5bca2c84e5e: Download complete
+4b299c049e4f: Verifying Checksum
+4b299c049e4f: Download complete
+91e158461d4c: Verifying Checksum
+91e158461d4c: Download complete
+0526f57f6bb3: Verifying Checksum
+0526f57f6bb3: Download complete
+6bda00aab163: Verifying Checksum
+6bda00aab163: Download complete
+599370ebdfdb: Download complete
+3debe21df7b8: Verifying Checksum
+3debe21df7b8: Download complete
+bd6da2b7ad00: Download complete
+3111ae43df8e: Verifying Checksum
+3111ae43df8e: Download complete
+4642a6d46aeb: Verifying Checksum
+4642a6d46aeb: Download complete
+dd2003bd419e: Verifying Checksum
+dd2003bd419e: Download complete
+d95fd6a8326e: Verifying Checksum
+d95fd6a8326e: Download complete
+49e25fdb34c3: Verifying Checksum
+f0f1cc16c840: Verifying Checksum
+f0f1cc16c840: Download complete
+f0f1cc16c840: Pull complete
+c5bca2c84e5e: Pull complete
+6f80705d1d37: Pull complete
+91e158461d4c: Pull complete
+4b299c049e4f: Pull complete
+4642a6d46aeb: Pull complete
+0526f57f6bb3: Pull complete
+6bda00aab163: Pull complete
+599370ebdfdb: Pull complete
+3debe21df7b8: Pull complete
+bd6da2b7ad00: Pull complete
+3111ae43df8e: Pull complete
+dd2003bd419e: Pull complete
+49e25fdb34c3: Pull complete
+d95fd6a8326e: Pull complete
+Digest: sha256:7aa3dad185d43fcf3d32abdf62fec86dab31b5159d66b2228275edc702181bb1
+Status: Downloaded newer image for quay.io/fenicsproject/stable:current
+[docker start 58acae9c63901a7fd0dee5170b51d6bfb81917da98430cc67373cfe097ce4960]

+58acae9c63901a7fd0dee5170b51d6bfb81917da98430cc67373cfe097ce4960
+After calling 'plt.show()' you can access matplotlib plots at http://localhost:3000
+[docker attach 58acae9c63901a7fd0dee5170b51d6bfb81917da98430cc67373cfe097ce4960]

+# FEniCS stable version image

+Welcome to FEniCS/stable!

+This image provides a full-featured and optimized build of the stable
+release of FEniCS.

+To help you get started this image contains a number of demo
+programs. Explore the demos by entering the 'demo' directory, for
+example:

+   cd ~/demo/python/documented/poisson
+   python3 demo_poisson.py

```

Pour sortir du container :
```
exit
```

Un fois revenu à la machine hôte, on peut lister les containers :
```diff
docker ps --all
+CONTAINER ID        IMAGE                                  COMMAND                  CREATED             STATUS                     PORTS               NAMES
+0121eaf771de        quay.io/fenicsproject/stable:current   "/sbin/my_init --qui…"   4 minutes ago       Exited (0) 2 minutes ago                       nifty_sanderson
+a8f846d82c74        hello-world                            "/hello"                 4 hours ago         Exited (0) 4 hours ago                         practical_banzai
```


Le container `0121eaf771de` crée lors du téléchargement de l'image FENICS n'est pas opérationnel, notamment pour les graphiques, car il aurait fallu le lancer avec certaines options. Nous pouvons donc le supprimer.... 

```
docker rm 0121eaf771de
```

Pour pouvoir afficher des graphiques au sein du container avec matplotlib, avant de lancer le container, il faut autoriser les application X11 :
```
xhost +
```
C'est expliqué dans la [documentation](https://fenics.readthedocs.io/projects/containers/en/latest/work_flows.html#use-graphical-applications-on-linux-hosts)

Pour construire un container qui affiche les images au sein d'un serveur web et pouvoir bien tester Fenics, il faut lancer la commande suivante, vu que l'on sait que l'on a téléchargé une image du projet Fenics appelée `quay.io/fenicsproject/stable:current` :
```
docker run -ti -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -p 127.0.0.1:8000:8000 quay.io/fenicsproject/stable:current
```
les graphiques seront alors disponibles à l'adresse :

http://127.0.0.1:8000

Une fois qu'on a arrêté un container, pour le relancer :
```
docker start 0121eaf771de
```

Pour se connecter à un container :

```
docker attach 0121eaf771de
```
Pour avoir des infos sur les containers, savoir où ils sont stockés dans l'arborescence :
```
docker info
```
Sur ubuntu, les containers sont dans `/var/lib/docker`. Pour parcourir ce répertoire, il peut être nécessaire de disposer de permissions root :
```
sudo nautilus &
```
