#!/usr/bin/env python
# This script receives OSC messages on port 1234 to render status about the muse
# on a Pimironi Blinkt LED strip.
# (connection information comes from muse-reconnect-status and muse-sock-osc-status.py
# Signal quality (HSI - headband status indicator) data comes from a separate
# signal processing script

import colorsys
import time

import blinkt
import liblo, sys
import os
# create server, listening on port 1234
mode = 0
try:
    server = liblo.Server(1234)
except liblo.ServerError, err:
    print str(err)
    sys.exit()

def musesock_callback(path, args):
    i,f = args
    #print "received message '%s' with arguments '%d' and '%f'" % (path, i, f)
    if (i==0):
        blinkt.set_pixel(0, 0, 0, 255)
    elif (i==1):
        blinkt.set_pixel(0, 255, 0, 255)
    elif (i==2):
        blinkt.set_pixel(0, int(255*(1.0-f)), int(255*f), 0)
    blinkt.show()

def reconnect_callback(path, args):
    #print "received message '%s' " % (path)
    blinkt.set_pixel(0, 1.0,1.0, 1.0)
    blinkt.show()

def hsi_callback(path, args):
    global mode
    h0,h1,h2,h3 = args
    #print "hsiVals %f %f %f %f'" % (h0,h1,h2,h3)
    if (h0<=254):
       blinkt.set_pixel(0, 0, int(255-h0), 0)
    else:
       blinkt.set_pixel(0, 0, 1, 0)
    if (h0<=1000):
       blinkt.set_pixel(1, 0, int(255-h0/4), 0)
    else:
       blinkt.set_pixel(1, 0, 1, 0)

    if (h1<=254):
       blinkt.set_pixel(2, int(255-h1), int((255-h1)/2), 0)
    else:
       blinkt.set_pixel(2, 1, 1, 0)
    if (h1<=1000):
       blinkt.set_pixel(3, int(255-h1/4), int((255-h1/4)/2), 0)
    else:
       blinkt.set_pixel(3, 1, 1, 0)

    if (h2<=254):
       blinkt.set_pixel(4, int(255-h2),0, 0)
    else:
       blinkt.set_pixel(4, 1, 0, 0)
    if (h2<=1000):
       blinkt.set_pixel(5, int(255-h2/4),0, 0)
    else:
       blinkt.set_pixel(5, 1, 0, 0)

    if (h3<=254):
       blinkt.set_pixel(6, 0, 0,int(255-h3))
    else:
       blinkt.set_pixel(6, 0, 0, 1)
    if (h3<=1000):
       blinkt.set_pixel(7, 0, 0, int(255-h3/4))
    else:
       blinkt.set_pixel(7, 0, 0, 1)
    blinkt.show()



def mode_callback(path, args):
    global mode
    i = args
    print "switching display mode to %d'" % (i)
    mode = i

def fallback(path, args, types, src):
    print "got unknown message '%s' from '%s'" % (path, src.url)
    for a, t in zip(args, types):
        print "argument of type '%s': %s" % (t, a)

# register method taking an int and a float
server.add_method("/muse-sock", "if", musesock_callback)
server.add_method("/muse-reconnect", None, reconnect_callback)
server.add_method("/hsi", "ffff", hsi_callback)
#server.add_method("/breath", "f", breath_callback)
#server.add_method("/heart", "f", heart_callback)
#server.add_method("/calm", "f", calm_callback)
#server.add_method("/mode", "i", mode_callback)
#server.add_method(None, None, fallback)

#light patterns
#LED 1
#muse-sock => connecting=purple,  connected=red->green depending on quality,
#muse-reconnect => sleeping=blue
#LED 2
#Blink every second if can ping Data proc computer.

#8 LEDs,   4hsi. 1 breath,  1 heart, 1 calm,  1 blink,    1 status,

# register a fallback for unhandled messages


blinkt.set_clear_on_exit()
blinkt.set_brightness(0.1)

# loop and dispatch messages every 100ms

while True:
    server.recv(100)
#hostname = "google.com"
#response = os.system("ping -c 1 " + hostname)
#if response == 0:
#    pingstatus = "Network Active"
#else:
#    pingstatus = "Network Error"



#spacing = 360.0 / 16.0
#hue = 0


#while True:
#    hue = int(time.time() * 100) % 360
#    for x in range(blinkt.NUM_PIXELS):
#        offset = x * spacing
#        h = ((hue + offset) % 360) / 360.0
#        r, g, b = [int(c*255) for c in colorsys.hsv_to_rgb(h, 1.0, 1.0)]
#        blinkt.set_pixel(x, r, g, b)
#    blinkt.show()    time.sleep(0.001)
