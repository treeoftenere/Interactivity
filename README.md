# Interactivity and Sensor/Effector integration 
A set of scripts to integrate streaming data with LXStudio over OSC for controlling the Tree of Tenere


This project aims to provide a reference system and suite of sensor integrations for TENERE,  written mostly in Python.  The initial platform involves the use of a Raspberry Pi (3 model B - tested) to receive streaming raw data from various sensors, including:

Works on PC/Mac or Raspberry Pi
* Muse headband (https://choosemuse.com, Bluetooth LE)
* USB microphone


Raspberry Pi specific parts list:
* Heartbeat sensor (http://pulsesensor.com, analog input)
* Grove expansion shield (https://www.dexterindustries.com/grovepi/, I2C)
* Pimoroni Blinkt (https://shop.pimoroni.com/products/blinkt, SPI) - equivalent to 1 of Tenere's leaves
* Grove accelerometer (http://wiki.seeed.cc/Grove-3-Axis_Digital_Accelerometer-16g/, I2C)
* Grove oled display (http://wiki.seeed.cc/Grove-OLED_Display_0.96inch/, I2C)
* Grove I2C Hub (http://wiki.seeed.cc/Grove-I2C_Hub/)
* Grove Button (https://www.seeedstudio.com/Grove-Button-p-766.html)
* USB Microphone (https://kinobo.co.uk/)


![TenerePi](/Images/tenere-raspberrypi-reference.png)


# Sensor Integration
The following outlines the installation process for a Raspberry Pi 3 using the latest version of the Raspian image (July 2017):

## Setting up the Raspberry Pi

* Start by following the installation instructions for downloading and writing the raspian image to a SD Card: https://www.raspberrypi.org/documentation/installation/installing-images/
* The Raspian Jessie with Deskop image has been tested: https://www.raspberrypi.org/downloads/raspbian/
* For Tenere, we suggest first configuring various options.  Be sure to turn on I2C, SPI, set your locale, keyboard layout, setup network, etc.  Please follow the relevant guides here: https://www.raspberrypi.org/documentation/configuration/
* Don't forget to change the default password to something more secure (use the `raspi-conf` tool)!!!
* Now, update all of the base packages and restart:

```
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y install fail2ban
sudo shutdown -r now
```

From your home directory (`/home/pi`), let's create a directory to hold all of our software:
```
cd
mkdir SOFTWARE
cd SOFTWARE
```

At the end of the tutorial, you should have a directory structure that looks something like this:
```
pi@raspberrypi:~/SOFTWARE $ ls
GrovePI
grovepi-zero
liblsl
Interactivity
Pimoroni
Tenere
pi@raspberrypi:~/SOFTWARE $
```

* Next, let's install the libraries we are going to use and clone any additional repositories (you may not need all of these for your specific setup, this tutorial includes everything for our reference system):

```
sudo apt-get -y install vim nano git git-core cmake python-pip python-dev
```

## Raspberry Pi accessories

* Install relevant libraries to enable the Blinkt LED strip
```
sudo apt-get -y install python-rpi.gpio python3-rpi.gpio
sudo apt-get -y install python-psutil python3-psutil python-tweepy
sudo apt-get -y install pimoroni python-blinkt python3-blinkt
cd ~/SOFTWARE
mkdir Pimoroni
cd Pimoroni
git clone https://github.com/pimoroni/blinkt.git
cd library
sudo python setup.py install
```

* Install relevant libraries for the Grove Pi expansion board (https://www.dexterindustries.com/GrovePi/get-started-with-the-grovepi/setting-software/)
```
sudo apt-get -y install libi2c-dev python-serial i2c-tools python-smbus python3-smbus arduino minicom
cd ~/SOFTWARE
git clone https://github.com/DexterInd/GrovePi.git
cd GrovePi/Script
sudo chmod +x install.sh
sudo ./install.sh
cd ~/SOFTWARE
https://github.com/initialstate/grovepi-zero.git
```

## Voice control with Jasper
```
cd ~/SOFTWARE/Interactivity/voicecontrol
```
Please follow the instructions at https://github.com/treeoftenere/Interactivity/voicecontrol


## Muse headband 

* For Muse Integration, several libraries are required.  Please note, this setup is only valid for the Muse 2016 (or later) versions:
```
sudo apt-get -y install python-liblo python-matplotlib python-numpy python-scipy python3-scipi python-seaborn liblo-tools
sudo pip install pygatt 
sudo pip install bitstring 
sudo pip install pexpect 
sudo pip install pylsl 
```

Unfortunately, there is an error in the latest pylsl package that distributes a library that is compiled for the wrong architecture.  We can fix this with the following:

```
cd ~/SOFTWARE
mkdir liblsl
cd liblsl
wget http://sccn.ucsd.edu/pub/software/LSL/SDK/liblsl-C-C++-1.11.zip
unzip liblsl-C-C++-1.11.zip
sudo cp liblsl-bcm2708.so /usr/local/lib/python2.7/dist-packages/pylsl-1.10.5-py2.7.egg/pylsl/liblsl32.so
```

Now turn on your Muse and let's figure out its network address
```
sudo hcitool lescan
```

You should see something like (please write down the hex address as we will use it later to connect):
```
00:55:DA:BO:0B:61 Muse-OB61
```

Now let's get the Muse talking to LXStudio.  This assumes you have the latest version of LXStudio running somewhere on your local network.  That is, we can test the Muse with LXStudio running on the same computer as the Muse is connecting.  However, our preference is to have the Muse stream data to the Raspberry Pi and then have the Pi send this data over a network to a show control computer (typically a Mac or PC) dedicated to running LXStudio. 

To do this, first grab the latest version of Processing and install for your desired platform (https://processing.org/download/) 

Then clone the lastest version of LXStudio (see more at: https://github.com/treeoftenere/Tenere)
```
git clone https://github.com/treeoftenere/Tenere.git
```

To get data from the Muse, we first use the Lab Streaming Layer library (previously installed, https://github.com/sccn/labstreaminglayer) to connect to the Muse over Bluetooth LE.  We then have a script that reads the streaming messages from LSL and then converts them to a format appropriate for OSC (http://opensoundcontrol.org/).  The `liblo` python package then takes care of streaming this newly processing sensor stream in OSC format to our show computer running LXStudio.

To test, let's clone this repository and launch our sensor processing pipeline (a big shout-out to @brainwaves for creating this):
```
cd ~/SOFTWARE
git clone https://github.com/treeoftenere/Interactivity
cd Interactivity
cd muse-sock
```

Now using the address we discovered previously, start the script that connects to the Muse (replace `00:55:DA:BO:0B:61` with the address of your Muse Headband):
```
python muse-sock.py --address 00:55:DA:BO:0B:61
```

Then in a second terminal, start our script for OSC streaming to LXStudio (replace `192.168.0.50` with the IP address of the machine where you are running Tenere's LXStudio:
```
cd ~/SOFTWARE/Interactivity/muse-sock
python muse-listener.py --oscip 192.168.0.50
```

Congratulations, you are now controlling Tenere with your brainwaves!!!!
