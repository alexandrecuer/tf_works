# wiringPi

related to the Gordon Henderson arduino wiring like library for the raspberry

## deb and sources

http://archive.ubuntu.com/ubuntu/pool/universe/w/wiringpi/

http://wiringpi.com/download-and-install/

## installation

cf http://wiringpi.com/wiringpi-updated-to-2-52-for-the-raspberry-pi-4b/
```
wget https://project-downloads.drogon.net/wiringpi-latest.deb
sudo dpkg -i wiringpi-latest.deb
```

` gpio -v` should return :
```
gpio version: 2.52
Copyright (c) 2012-2018 Gordon Henderson
This is free software with ABSOLUTELY NO WARRANTY.
For details type: gpio -warranty

Raspberry Pi Details:
  Type: Pi 3, Revision: 02, Memory: 1024MB, Maker: Sony 
  * Device tree is enabled.
  *--> Raspberry Pi 3 Model B Rev 1.2
  * This Raspberry Pi supports user-level GPIO access.
```
`gpio readall` should return :
```
 +-----+-----+---------+------+---+---Pi 3B--+---+------+---------+-----+-----+
 | BCM | wPi |   Name  | Mode | V | Physical | V | Mode | Name    | wPi | BCM |
 +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+
 |     |     |    3.3v |      |   |  1 || 2  |   |      | 5v      |     |     |
 |   2 |   8 |   SDA.1 | ALT0 | 1 |  3 || 4  |   |      | 5v      |     |     |
 |   3 |   9 |   SCL.1 | ALT0 | 1 |  5 || 6  |   |      | 0v      |     |     |
 |   4 |   7 | GPIO. 7 |   IN | 0 |  7 || 8  | 1 | ALT0 | TxD     | 15  | 14  |
 |     |     |      0v |      |   |  9 || 10 | 1 | ALT0 | RxD     | 16  | 15  |
 |  17 |   0 | GPIO. 0 |   IN | 0 | 11 || 12 | 0 | IN   | GPIO. 1 | 1   | 18  |
 |  27 |   2 | GPIO. 2 |   IN | 0 | 13 || 14 |   |      | 0v      |     |     |
 |  22 |   3 | GPIO. 3 |   IN | 0 | 15 || 16 | 0 | IN   | GPIO. 4 | 4   | 23  |
 |     |     |    3.3v |      |   | 17 || 18 | 0 | IN   | GPIO. 5 | 5   | 24  |
 |  10 |  12 |    MOSI |   IN | 0 | 19 || 20 |   |      | 0v      |     |     |
 |   9 |  13 |    MISO |   IN | 0 | 21 || 22 | 0 | IN   | GPIO. 6 | 6   | 25  |
 |  11 |  14 |    SCLK |   IN | 0 | 23 || 24 | 1 | IN   | CE0     | 10  | 8   |
 |     |     |      0v |      |   | 25 || 26 | 0 | IN   | CE1     | 11  | 7   |
 |   0 |  30 |   SDA.0 |   IN | 1 | 27 || 28 | 1 | IN   | SCL.0   | 31  | 1   |
 |   5 |  21 | GPIO.21 |   IN | 1 | 29 || 30 |   |      | 0v      |     |     |
 |   6 |  22 | GPIO.22 |   IN | 1 | 31 || 32 | 0 | IN   | GPIO.26 | 26  | 12  |
 |  13 |  23 | GPIO.23 |   IN | 0 | 33 || 34 |   |      | 0v      |     |     |
 |  19 |  24 | GPIO.24 |   IN | 0 | 35 || 36 | 0 | IN   | GPIO.27 | 27  | 16  |
 |  26 |  25 | GPIO.25 |   IN | 0 | 37 || 38 | 0 | IN   | GPIO.28 | 28  | 20  |
 |     |     |      0v |      |   | 39 || 40 | 0 | IN   | GPIO.29 | 29  | 21  |
 +-----+-----+---------+------+---+----++----+---+------+---------+-----+-----+
 | BCM | wPi |   Name  | Mode | V | Physical | V | Mode | Name    | wPi | BCM |
 +-----+-----+---------+------+---+---Pi 3B--+---+------+---------+-----+-----+
```
`
## exploring

```
$ ar tv wiringpi-latest.deb 
rw-r--r-- 0/0      4 Jun 23 20:11 2019 debian-binary
rw-r--r-- 0/0    560 Jun 23 20:11 2019 control.tar.xz
rw-r--r-- 0/0  51508 Jun 23 20:11 2019 data.tar.xz
```

```
$ ar x wiringpi-latest.deb 
$ ls
control.tar.xz  debian-binary  wiringpi-latest.deb  data.tar.xz 
```

```
tar -xvf data.tar.xz
./
./usr/
./usr/bin/
./usr/bin/gpio
./usr/include/
./usr/include/ads1115.h
./usr/include/bmp180.h
./usr/include/drcNet.h
./usr/include/drcSerial.h
./usr/include/ds1302.h
./usr/include/ds18b20.h
./usr/include/gertboard.h
./usr/include/htu21d.h
./usr/include/lcd.h
./usr/include/lcd128x64.h
./usr/include/max31855.h
./usr/include/max5322.h
./usr/include/maxdetect.h
./usr/include/mcp23008.h
./usr/include/mcp23016.h
./usr/include/mcp23016reg.h
./usr/include/mcp23017.h
./usr/include/mcp23s08.h
./usr/include/mcp23s17.h
./usr/include/mcp23x08.h
./usr/include/mcp23x0817.h
./usr/include/mcp3002.h
./usr/include/mcp3004.h
./usr/include/mcp3422.h
./usr/include/mcp4802.h
./usr/include/pcf8574.h
./usr/include/pcf8591.h
./usr/include/piFace.h
./usr/include/piGlow.h
./usr/include/piNes.h
./usr/include/pseudoPins.h
./usr/include/rht03.h
./usr/include/scrollPhat.h
./usr/include/sn3218.h
./usr/include/softPwm.h
./usr/include/softServo.h
./usr/include/softTone.h
./usr/include/sr595.h
./usr/include/wiringPi.h
./usr/include/wiringPiI2C.h
./usr/include/wiringPiSPI.h
./usr/include/wiringSerial.h
./usr/include/wiringShift.h
./usr/include/wpiExtensions.h
./usr/lib/
./usr/lib/libwiringPi.so.2.52
./usr/lib/libwiringPiDev.so.2.52
./usr/share/
./usr/share/man/
./usr/share/man/man1/
./usr/share/man/man1/gpio.1
./usr/lib/libwiringPi.so
./usr/lib/libwiringPiDev.so
```
