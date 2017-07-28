# muse-sock

This directory contains several utilities for streaming Muse data in real-time to LXStudio over LSL and OSC

muse-sock now supports EEG,  accelerometer and gyro data from the muse

muse-sock.py has a watchdog exits if data hasn’t been received for 5s, or too much data is lost over 10s

debug messages come out once per second and report approx time since last accelerometer data packet,  percentage of data lost over the last second, and average data loss over the last 10s


Run these on the command line for a short description of the functionality of each script:

```
python muse-sock.py --help
python muse-listener.py --help
```

## Runtime
Start both `muse-sock.py` and `muse-listener.py` on the pi to receive,  parse, and send out OSC messages.

Optionally:
muse-reconnect is a bash script that tries to reconnect to a muse forever


