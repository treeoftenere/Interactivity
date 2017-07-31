# Streaming data from the Raspberry Pi

This folder contains scripts for gathering data from the Raspberry Pi and then streaming it to LX over OSC (Open Sound Control, http://opensoundcontrol.org/).


For a description of the default settings and a list of command line options, please try:
```
python grove-osc.py --help
```

Typical usage is:
```
python grove-osc.py --oscip <IP of machine running LX>
```

Presently this script will collect data from whatever is connected to the 3 analog and 1 digital connector on the Grove Pi Zero board. 

For use with the Tenere reference system:
* The Pulse Sensor is plugged into A0
* The Grove Button is connected to D0
* The I2C Hub is connected to the I2C port
* The Grove Accelerometer and OLED display are both connected to the I2C Hub.


If you would also like to use the OLED for a menu system or to display data during debugging, for example, we have provided `grove-oled.py`.  For more information, please try:
```
python grove-oled.py --help
```

