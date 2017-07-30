import time
import numpy as np
import grovepi
import grove_oled
from adxl345 import ADXL345


usage = "python sensor-osc.py --oscip 127.0.0.1 --oscport 7878 --with_accel 1 --with_oled 0 --with_pulse 1"
parser = OptionParser(usage=usage)
parser.add_option("-a", "--with_accel", 
				  dest="withAccel", type='int', default=1,
				  help="Is a Grove Accelerometer ADXL345 connected via I2C?")
parser.add_option("-d", "--debug", 
                  dest="printDebug", type='int', default=0,
                  help="Print the sensor values to stdout?")

global axes

print("starting display")
grove_oled.oled_init()
grove_oled.oled_clearDisplay()
grove_oled.oled_setNormalDisplay()
grove_oled.oled_setVerticalMode()



while True:
	if options.withAccel:	

		time.sleep(0.2) # only update as often as necessary
		
		axes = adxl345.getAxes(True)

		if options.printDebug:
			print("accel: x = %.3fG, y = %.3fG, z = %.3fG" % ( axes['x'], axes['y'], axes['z']))		
		
        grove_oled.oled_clearDisplay()
		grove_oled.oled_setTextXY(0,0)
		grove_oled.oled_putString("x = %.3fG" % (axes['x']))
		grove_oled.oled_setTextXY(1,0)
		grove_oled.oled_putString("y = %.3fG" % (axes['y']))
		grove_oled.oled_setTextXY(2,0)
		grove_oled.oled_putString("z = %.3fG" % (axes['z']))
		grove_oled.oled_setTextXY(3,0)




