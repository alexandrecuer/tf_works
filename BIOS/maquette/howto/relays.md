# how to operate relays

![relay 12V 10A 230V](img_relays/songle12Vrelay10A.jpg)

Le relais s’alimente et se commande en 12V (SRD-12VDC-SL-C)

On peut lire dessus qu’on peut faire passer :

- 10 ampères sous 250V alternatif (50 ou 60Hz), c’est-à-dire du secteur en France.
- 10 ampères sous 125V alternatif (50 ou 60Hz), c’est-à-dire environ 1KW pour les pays en 110V. 
- 10 ampères en 28 ou 30V continu.

Cela permet de commander un appareil jusqu’à environ 2KW, soit une ampoule ou un radiateur. 

Attention aux moteurs qui ont souvent un très fort appel de courant au démarrage :

- Impossible de brancher la machine à laver ou le frigo sur ce genre de relais. 
- Il semble possible de contrôler un ventilateur dont le moteur fait quelques dizaine de watt.

## relays

![relays](img_relays/relays.jpg)

![relay power supply](img_relays/power_supply.jpg)

## equipement to control

![power cable of the equipment to control](img_relays/equipment_to_control1.jpeg)

![the lamp](img_relays/lamp.jpg)

![connect the equipment to the relay](img_relays/equipment_to_control2.jpeg)

## raspberry control

![control with raspberry](img_relays/RPI.jpg)

To check pinout on the raspberry : https://pinout.xyz/#

```
import RPi.GPIO as GPIO
import curses

GPIO.setmode(GPIO.BCM)

nb=26

GPIO.setup(nb, GPIO.OUT)
GPIO.output(nb, GPIO.HIGH)

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
stdscr.addstr(0,0, "a=allumer s=stop q=quitter")

while True:
    c = stdscr.getch()
    if c == ord('a'):
        GPIO.output(nb, GPIO.HIGH)
    if c == ord('s'):
        GPIO.output(nb, GPIO.LOW)
    if c == ord('q'):
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()
        GPIO.cleanup()
        break
```
more about curses : https://docs.python.org/3/howto/curses.html

more about RPi.GPIO : https://pypi.org/project/RPi.GPIO/

Note that RPi.GPIO is unsuitable for real-time or timing critical applications. This is because you can not predict when Python will be busy garbage collecting. It also runs under the Linux kernel which is not suitable for real time applications - it is multitasking O/S and another process may be given priority over the CPU, causing jitter in your program. If you are after true real-time performance and predictability, buy yourself an Arduino

https://raspberrypi.pagesperso-orange.fr/dossiers/26-33.htm
