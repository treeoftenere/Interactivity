import bitstring
import pygatt
import numpy as np
from time import time
from sys import platform
import socket
import sys

class Muse(object):
    """Muse 2016 headband or SmithX prototype
    """
    def __init__(self, address, callback, host, port, device_type=None, 
                 backend='auto', interface=None):
        self.address = address
        self.callback = callback
        self.device_type = device_type

        self.interface = interface
       
        self.HOST = host #'192.168.1.118'
        self.PORT = port #9999
        self.s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if backend == 'auto':
            if platform == "linux" or platform == "linux2":
                self.backend = 'gatt'
            else:
                self.backend = 'bgapi'
        elif backend in ['gatt', 'bgapi']:
            self.backend = backend
        else:
            raise(ValueError('Backend must be auto, gatt or bgapi'))

    def connect(self, interface=None):
        """Connect to the device

        Args:
            interface: if specified, call the backend with this interface
        """
        if self.backend == 'gatt':
            self.interface = self.interface or 'hci0'
            self.adapter = pygatt.GATTToolBackend(self.interface)
        else:
            self.adapter = pygatt.BGAPIBackend(serial_port=self.interface)

        self.adapter.start()
        self.device = self.adapter.connect(self.address, timeout=5)

        self._subscribe_eeg()
        self._subscribe_acc()
        self._subscribe_gyro()


    def start(self):
        """Start streaming."""
	
        if self.device_type.lower() is ('muse'):
            # Change preset to 31
            self.device.char_write_handle(0x00e,
                                          [0x04, 0x50, 0x33, 0x31, 0x0a],
                                          False)
        self.device.char_write_handle(0x000e, [0x02, 0x64, 0x0a], False)

    def stop(self):
        """Stop streaming."""
        self.device.char_write_handle(0x000e, [0x02, 0x68, 0x0a], False)

    def disconnect(self):
        """Disconnect."""
        self.device.disconnect()
        self.adapter.stop()

    def _subscribe_eeg(self):
        """Subscribe to EEG stream."""
        self.device.subscribe('273e0003-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0004-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0005-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0006-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0007-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)

    def _subscribe_acc(self):
        """Subscribe to the accelerometer stream."""
        self.device.subscribe('273e000a-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_acc)

    def _subscribe_gyro(self):
        """Subscribe to the gyroscope stream."""
        self.device.subscribe('273e0009-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_gyro)

    def _handle_eeg(self, handle, data):
        self.s.sendto("%s%s"%(handle,data), (self.HOST,self.PORT))

    def _handle_acc(self, handle, data):
#        print("acc ", handle)
        self.s.sendto("%s%s"%(handle,data), (self.HOST,self.PORT))
        self.callback()

    def _handle_gyro(self, handle, data):
#        print("gyro ", handle)
        self.s.sendto("%s%s"%(handle,data), (self.HOST,self.PORT))
