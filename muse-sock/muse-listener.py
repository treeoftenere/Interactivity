import socket
import bitstring
import numpy as np
import liblo as lo
from optparse import OptionParser

usage = "python muse-listener.py --port 9999 --oscip 127.0.0.1 --oscport 7878"
parser = OptionParser(usage=usage)
parser.add_option("-l", "--port", 
                  dest="port", type='int', default=9999,
                  help="port to listen to")
parser.add_option("-o", "--oscip", 
                  dest="oscip", type='string', default="127.0.0.1",
                  help="IP address to send OSC message to")
parser.add_option("-p", "--oscport", 
                  dest="oscport", type='int', default=7878,
                  help="host port")
(options, args) = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
sock.bind(('', options.port))
outputAddress = lo.Address(options.oscip, options.oscport)
eegdata = np.zeros((5,12))
gyrodata = np.zeros((3,3))
accdata = np.zeros((3,3))
timestamps = np.zeros(12)
while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    """
        samples are received in this order : 44, 41, 38, 32, 35
        wait until we get 35 and call the data
        """
    handle = int(data[0:2])
    #print(handle)
    aa = bitstring.Bits(bytes=data[2:])

    if (handle==20):
        pattern = "uint:16,int:16,int:16,int:16,int:16,int:16,int:16,int:16,int:16,int:16"
        res = aa.unpack(pattern)
        timestamp = res[0]
        data = res[1:]
        # 16 bits on a 256 dps range
        data = np.array(data)*0.0074768
        gyrodata[0] = data[0:3]
        gyrodata[1] = data[3:6]
        gyrodata[2] = data[6:9]
        for ind in range(3):
                gyroMessage = lo.Message('/muse/gyro', gyrodata[ind][0], gyrodata[ind][1],gyrodata[ind][2])
                lo.send(outputAddress, gyroMessage)

    elif (handle==23):
        pattern = "uint:16,int:16,int:16,int:16,int:16,int:16,int:16,int:16,int:16,int:16"
        res = aa.unpack(pattern)
        timestamp = res[0]
        data = res[1:]
        # 16 bits on a 256 dps range
        data = np.array(data)*0.0074768
        accdata[0] = data[0:3]
        accdata[1] = data[3:6]
        accdata[2] = data[6:9]
        for ind in range(3):
                accMessage = lo.Message('/muse/acc', accdata[ind][0], accdata[ind][1],accdata[ind][2])
                lo.send(outputAddress, accMessage)

    else:
      pattern = "uint:16,uint:12,uint:12,uint:12,uint:12,uint:12,uint:12, \
                 uint:12,uint:12,uint:12,uint:12,uint:12,uint:12"
      res = aa.unpack(pattern)
      timestamp = res[0]
      data = res[1:]
      data = 0.40293 * np.array(data)
    
      #print(int(handle)," ",  timestamp," ", data)
      #print(handle)
    
      index = int((handle - 32) / 3)
      eegdata[index] = data
      timestamps[index] = timestamp
      if handle == 35:
            for ind in range(12):
                    eegMessage = lo.Message('/muse/eeg', eegdata[0][ind], eegdata[1][ind],
                                  eegdata[2][ind], eegdata[3][ind],
                                  eegdata[4][ind])
                    lo.send(outputAddress, eegMessage)
    
