"""Connect to a Muse and send its data through the LSL protocol
"""

from muse import Muse
from time import sleep
from optparse import OptionParser
import time

now = time.time()

usage = "python muse-socket.py --address  00:55:DA:B0:32:B1 --host 192.168.1.118 --port 9999"
parser = OptionParser(usage=usage)
parser.add_option("-a", "--address",
                  dest="address", type='string', default="00:55:DA:B0:06:D6",
                  help="Device mac address.")
parser.add_option("-i", "--host",
                  dest="host", type='string', default="127.0.0.1",
                  help="host IP")
parser.add_option("-p", "--port",
                  dest="port", type='int', default=9999,
                  help="host port")
parser.add_option("-b", "--backend",
                  dest="backend", type='string', default="auto",
                  help="Pygatt backend to use. Can be auto, gatt or bgapi")
parser.add_option("-d", "--device-type",
                  dest="device_type", type='string', default="muse",
                  help="Device type.")

(options, args) = parser.parse_args()

countACC = 0;
def process():
    global now
    global countACC
    now = time.time()
    countACC+=1.0

muse = Muse(address=options.address, device_type=options.device_type,
            host=options.host, port=options.port,
            callback=process, backend=options.backend,interface=None)

muse.connect()
print('Connected')
muse.start()
print('Streaming')
idx =0
losshist =[0 for i in range(10)]
while 1:
    try:
        sleep(1)
        dataloss =max(0.0,100.0-countACC*3/50*100.0)
        losshist[idx] = dataloss
        idx=(idx+1)%10
        avgloss =sum(losshist)/float(len(losshist))
        print('waited: %2f' % (time.time()-now),  'dataloss: %.1f' % dataloss,'avgloss: %f' % avgloss )
        countACC = 0;
        if ((time.time()-now)>500):
            break
        if ((avgloss>40)):
            break
    except:
        break

muse.stop()
muse.disconnect()
