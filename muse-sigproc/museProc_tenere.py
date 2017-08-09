#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 10:44:58 2017

@author: chris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Muse output server
==================

This script shows how to process and stream different Muse outputs, such as:
-

TODO:
    - Make musetools.realtime.EEGServer more general so we don't have to
      create a new class for this example (and instead only plug in
      necessary pieces or inherit from it).

"""

import time
from threading import Thread

import numpy as np
from scipy import signal, interpolate
from pylsl import StreamInlet, resolve_byprop
USE_LIBLO = True  # s.name != 'nt'
if USE_LIBLO:
    from liblo import ServerThread, Address, Message, Bundle, send
    print('using Liblo')
else:
    from pythonosc import dispatcher, osc_server, udp_client
    print('using pythonOSC')

import live_utils as ut


from optparse import OptionParser

usage = "python museProc_tenere.py --port 9999 --oscip 127.0.0.1 --oscport 7878 --sparseip 10.0.0.14 --sparseport 1234"
parser = OptionParser(usage=usage)
parser.add_option("-l", "--port",
                  dest="port", type='int', default=9810,
                  help="port to listen for muse data on")
parser.add_option("-o", "--oscip",
                  dest="oscip", type='string', default="127.0.0.1",
                  help="IP address of Tenere LXstudio to send OSC message to")
parser.add_option("-p", "--oscport",
                  dest="oscport", type='int', default=7878,
                  help="The oort that Tenere LXstudio is listening on ")
parser.add_option("-r", "--sparseip",
                  dest="sparseip", type='string', default="127.0.0.1",
                  help="IP address of the Pi to send OSC status message to")
parser.add_option("-s", "--sparseport",
                  dest="sparseport", type='int', default=1234,
                  help="Port for OSC status messages on the Pi")


(options, args) = parser.parse_args()


class FFTServer():
    """Server to receive EEG data and stream classifier outputs.

    Attributes:
        See args.

    Args:
        incoming (str or dict): incoming data stream. If provided as a
            string, look for an LSL stream with the corresponding type. If
            provided as dict with fields `address` and `port`, open an OSC
            port at that address and port.
        outgoing (str or dict): outgoing data stream. If provided as a
            string, stream to an LSL stream with the corresponding type. If
            provided as dict with fields `address` and `port`, stream to an
            OSC port at that address and port.

    Keyword Args:
        config (dict): dictionary containing the configuration and
            preprocessing parameters, e.g. (these are the default values):

                config = {'fs': 256.,
                          'n_channels': 4,
                          'raw_buffer_len': 3 * fs,
                          'filt_buffer_len': 3 * fs,
                          'window_len': fs,
                          'step': int(fs / 10),
                          'filter': ([1], [1]),
                          'psd_window_len': 256.,
                          'psd_buffer_len': 10}

        device_source (str): Device from which the data is coming from.
            'muse' or 'vive'
        streaming_source (str): Software source of the data stream:
            'muselsl'
            'musedirect'
            'musemonitor'
        debug_outputs (bool): if True, send debug outputs (not used by VR
            experience)
        verbose (bool): if True, print status whenever new data is
            received or sent.

    """
    def __init__(self, incoming, outgoing,  sparseOutput=None, config={}, device_source='Muse',
                 software_source='muselsl', debug_outputs=True, verbose=False):

        self.incoming = incoming
        self.outgoing = outgoing
        self.sparseOutput = sparseOutput
        self.device_source = device_source
        self.software_source = software_source
        self.debug_outputs = debug_outputs
        self.verbose = verbose
        self.eeg_chunk_length = 12

        # 1. Initialize inlet
        if isinstance(self.incoming, str):  # LSL inlet
            print('Looking for the {} stream...'.format(incoming))
            self._stream = resolve_byprop('type', incoming, timeout=2)

            if len(self._stream) == 0:
                raise(RuntimeError('Can\'t find {} stream.'.format(incoming)))
            print('Aquiring data from the \'{}\' stream...'.format(incoming))

            self._inlet = StreamInlet(self._stream[0],
                                      max_chunklen=self.eeg_chunk_length)
            self._info_in = self._inlet.info()

        else:  # OSC port
            if USE_LIBLO:
                self._osc_server = ServerThread(incoming['port'])
                print('OSC server initialized at port {}.'.format(
                        incoming['port']))
            else:
                self._dispatcher = dispatcher.Dispatcher()
                print('python-osc dispatcher initialized.')

        # 2. Initialize outlets
        if not isinstance(self.outgoing, tuple):
            self.outgoing = [self.outgoing]
        self._output_threads = []
        for out in self.outgoing:

            if isinstance(out, str):  # LSL outlet
                raise NotImplementedError

            elif isinstance(out, dict):  # OSC port
                if USE_LIBLO:
                    self._output_threads.append(Address(out['address'],
                                                  out['port']))
                else:
                    raise NotImplementedError
#                    self._client = udp_client.SimpleUDPClient(
#                            outgoing['address'], outgoing['port'])
                print('OSC client initialized at {}:{}.'.format(
                            out['address'], out['port']))

        if (self.sparseOutput !=None):
            if not isinstance(self.sparseOutput, tuple):
                self.sparseOutput = [self.sparseOutput]
            self._sparseOutput_threads = []
            for out in self.sparseOutput:
                if isinstance(out, str):  # LSL outlet
                    raise NotImplementedError

                elif isinstance(out, dict):  # OSC port
                    if USE_LIBLO:
                        self._sparseOutput_threads.append(Address(out['address'],
                                                      out['port']))
                    else:
                        raise NotImplementedError
                    print('OSC sparse output client initialized at {}:{}.'.format(
                                out['address'], out['port']))


        # 3. Initialize internal buffers and variables
        self._init_processing(config)

    def _init_processing(self, config):
        """Initialize internal buffers and variables for EEG processing.

        Args:
            config (dict): dictionary containing various parameters. See
                DEFAULT_CONFIG below for default values.

                fs (float): sampling frequency
                n_channels (int): number of channels
                raw_buffer_len (int): raw data buffer length
                filt_buffer_len (int): filtered data buffer length
                window_len (int): processing window length
                step (int): number of samples between two consecutive
                    windows to process
                filter (tuple or dict): filtering parameters. If provided
                    as a tuple, the first and second elements should be
                    the `b` and `a` coefficients of a filter. If provided
                    as a dictionary, the fields `order`, `l_freq`, `h_freq`
                    and `method` are required; the function
                    pre.get_filter_coeff() will then be used to compute the
                    coefficients.
                    If None, don't use a filter (windows will be extracted
                    from the raw buffer).
                psd_window_len (int): length of the window to use for PSD
                psd_buffer_len (int): PSD buffer length
        """
        DEFAULT_CONFIG = {'fs': 256.,
                          'n_channels': 5,
                          'raw_buffer_len': 3 * 256,
                          'filt_buffer_len': 3 * 256,
                          'window_len': 256,
                          'step': int(256 / 10),
                          'filter': ([1], [1]),
                          'filter_bank': {},
                          'psd_window_len': 256.,
                          'psd_buffer_len': 10}

        config = {**DEFAULT_CONFIG, **config}

        self.fs = config['fs']
        self.n_channels = config['n_channels']

        # Initialize EEG channel remapping parameters
        self.eeg_ch_remap = None
        if self.device_source.lower() == 'vive':
            self.eeg_ch_remap = [3, 1, 2, 3, 4]
            self.n_channels = 5
        if self.software_source.lower() == 'musedirect':
            self.eeg_ch_remap[-1] = 5
            self.n_channels = 5
        if self.device_source.lower() == 'leroy':
            self.eeg_ch_remap = None
            self.n_channels = 4
        if self.device_source.lower() == 'muse':
            self.eeg_ch_remap = None
            self.n_channels = 4
        if self.device_source.lower() == 'vivehr':
            self.eeg_ch_remap = [3, 1, 2, 3, 0]
            self.n_channels = 5

        # Initialize the EEG buffers
        raw_buffer_len = int(config['raw_buffer_len'])
        filt_buffer_len = int(config['filt_buffer_len'])

        self.eeg_buffer = ut.NanBuffer(raw_buffer_len, self.n_channels)
        self.filt_eeg_buffer = ut.CircularBuffer(filt_buffer_len,
                                                 self.n_channels)
        self.hpfilt_eeg_buffer = ut.CircularBuffer(filt_buffer_len,
                                                 self.n_channels)
        self.smooth_eeg_buffer = ut.CircularBuffer(filt_buffer_len,
                                                 self.n_channels)
        self.eyeH_buffer = ut.CircularBuffer(100,1)

        # Initialize the EEG filter
        if config['filter']:
            if isinstance(config['filter'], tuple):
                b = config['filter'][0]
                a = config['filter'][1]
            elif isinstance(config['filter'], dict):
                b, a = ut.get_filter_coeff(self.fs, **config['filter'])
            zi = np.tile(signal.lfilter_zi(b, a), (self.n_channels, 1)).T
            self.bandpass_filt = {'b': b,
                                  'a': a,
                                  'zi': zi}
        if config['hpfilter']:
            b = config['hpfilter'][0]
            a = config['hpfilter'][1]
            zi = np.tile(signal.lfilter_zi(b, a), (self.n_channels, 1)).T
            self.hp_filt = {'b': b,
                            'a': a,
                            'zi': zi}
        if config['lpfilter']:
            b = config['lpfilter'][0]
            a = config['lpfilter'][1]
            zi = np.tile(signal.lfilter_zi(b, a), (self.n_channels, 1)).T
            self.lp_filt = {'b': b,
                            'a': a,
                            'zi': zi}

        # Initialize the filter bank
        if config['filter_bank']:
            self.filter_bank = {}
            for name, coeff in config['filter_bank'].items():
                zi = np.tile(signal.lfilter_zi(coeff[0], coeff[1]),
                                               (self.n_channels, 1)).T
                self.filter_bank[name] = {'b': coeff[0],
                                          'a': coeff[1],
                                          'zi': zi}


        # Initialize processing parameters
        self.window_len = int(config['window_len'])
        self.step = int(config['step'])

        # Initialize processing buffers
        psd_buffer_len = int(config['psd_buffer_len'])
        self.psd_buffer = ut.CircularBuffer(psd_buffer_len, 129,
                                            self.n_channels)

        # Initialize scoring histograms
        decayRate = 0.997
        self.hists = {'delta': ut.Histogram(1000, self.n_channels, bounds=(0, 50), min_count=80, decay=decayRate ),
                      'theta': ut.Histogram(1000, self.n_channels, bounds=(0, 30),min_count=80, decay=decayRate),
                      'alpha': ut.Histogram(1000, self.n_channels,bounds=(0, 20), min_count=80, decay=decayRate),
                      'beta': ut.Histogram(1000, self.n_channels,bounds=(0, 10), min_count=80, decay=decayRate),
                      'gamma': ut.Histogram(1000, self.n_channels,bounds=(0, 10), min_count=80, decay=decayRate)}
        self.eyeH_hist =  ut.Histogram(500, 1, bounds=(0, 10000), min_count=80, decay=decayRate )
        self.emg_hist =  ut.Histogram(500, 1, bounds=(0, 10), min_count=80, decay=decayRate )
        self.blinkwait = 0
        self.blink = 0
        self.firstWindowProc = True
        self.band_names =0
        self.band_powers =0
        self.ratio_powers=0
        self.ratio_names=0

        # Used for calm score
        self.slow_calm_score = 0
        self.slow_alpha_score = 0
        self.eye_mov_percent_buffer = ut.CircularBuffer(256, 1)
        self.slow_calm_score_buffer = ut.CircularBuffer(512, 1)
        self.increments_buffer = ut.CircularBuffer(512, 1)
        self.low_freq_chs_buffer = ut.CircularBuffer(150, 2)
        self.low_freq_chs_std = 1

        ######################################################################
        # BODY Motion Processing,  Accelerometer, Gyro

        raw_buffer_len = 150
        filt_buffer_len = 150
        self.acc_window_len = 50
        self.acc_buffer = ut.NanBuffer(raw_buffer_len, 3)
        self.filt0_buffer = ut.CircularBuffer(filt_buffer_len,3)
        self.heart_buffer = ut.CircularBuffer(150,1)
        self.breath_buffer = ut.CircularBuffer(500,1)

        # Initialize the Body Filters
        if config['filter0']:
            b = config['filter0'][0]
            a = config['filter0'][1]
            zi = np.tile(signal.lfilter_zi(b, a), (3, 1)).T
            self.filter0 = {'b': b,'a': a,'zi': zi}
        if config['filter1']:
            b = config['filter1'][0]
            a = config['filter1'][1]
            zi = np.tile(signal.lfilter_zi(b, a), (3, 1)).T
            self.filter1 = {'b': b,'a': a,'zi': zi}
        if config['filter2']:
            b = config['filter2'][0]
            a = config['filter2'][1]
            zi = signal.lfilter_zi(b, a)
            self.filter2 = {'b': b,'a': a,'zi': zi}
        if config['filter3']:
            b = config['filter3'][0]
            a = config['filter3'][1]
            zi = np.tile(signal.lfilter_zi(b, a), (3, 1)).T
            self.filter3 = {'b': b,'a': a,'zi': zi}
        if config['filter4']:
            b = config['filter4'][0]
            a = config['filter4'][1]
            zi = signal.lfilter_zi(b, a)
            self.filter4 = {'b': b,'a': a,'zi': zi}
        if config['filter5']:
            b = config['filter5'][0]
            a = config['filter5'][1]
            zi = signal.lfilter_zi(b, a)
            self.filter5 = {'b': b,'a': a,'zi': zi}
        if config['filter6']:
            b = config['filter6'][0]
            a = config['filter6'][1]
            zi = signal.lfilter_zi(b, a)
            self.filter6 = {'b': b,'a': a,'zi': zi}
        if config['filter7']:
            b = config['filter7'][0]
            a = config['filter7'][1]
            zi = signal.lfilter_zi(b, a)
            self.filter7 = {'b': b,'a': a,'zi': zi}



    def _update_eeg_liblo_osc(self, path, args):
        """Collect new EEG data point(s) from pyliblo OSC and process.

        Args:
            path (str): OSC path listened to
            args (list): received values
        """
        if self.verbose:
            print('Receiving OSC packet!')
        sample = np.array(args).reshape(1, -1)
        self._process_eeg(sample[:, :self.n_channels], 0)

    def _update_eeg_python_osc(self, unused_addr, args, *chs):
        """Collect new EEG data point(s) from python-osc and process.

        Args:
            path (str): OSC path listened to
            args (list): received values
        """
        if self.verbose:
            print('Receiving OSC packet!')
        sample = np.array(chs).reshape(1, -1)
        self._process_eeg(sample[:, :self.n_channels], 0)


    def _update_acc_liblo_osc(self, path, args):
        if self.verbose:
            print('Receiving ACC packet!')
        sample = np.array(args).reshape(1, -1)
        self._process_acc(sample[:, :3], 0)

    def _update_gyro_liblo_osc(self, path, args):
        if self.verbose:
            print('Receiving GYRO packet!')
        sample = np.array(args).reshape(1, -1)
        self._process_gyro(sample[:, :3], 0)


    def _process_eeg(self, samples, timestamp):
        """Process EEG.

        Process EEG. Includes buffering, filtering, windowing and pipeline.

        Args:
            samples (numpy.ndarray): new EEG samples to process
            timestamp (float): timestamp

        Returns:
            output (scalar): output of the pipeline
        """

        # Re-map
        if self.eeg_ch_remap:
            samples = samples[:, self.eeg_ch_remap]

        self.eeg_buffer.update(samples)
#        self._send_outputs(samples, timestamp, 'raw_eeg')

        # Apply filtes
        filt_samples = samples;

        if config['filter']:
            filt_samples, self.bandpass_filt['zi'] = signal.lfilter(
                    self.bandpass_filt['b'], self.bandpass_filt['a'],
                    samples, axis=0, zi=self.bandpass_filt['zi'])
            # self._send_filtered_eeg(filt_samples, timestamp)
        self.filt_eeg_buffer.update(filt_samples)

        if config['hpfilter']:
            filt_samples, self.hp_filt['zi'] = signal.lfilter(
                    self.hp_filt['b'], self.hp_filt['a'],
                    filt_samples, axis=0, zi=self.hp_filt['zi'])
        self.hpfilt_eeg_buffer.update(filt_samples)

        if config['lpfilter']:
            smooth_eeg_samples, self.lp_filt['zi'] = signal.lfilter(
                    self.lp_filt['b'], self.lp_filt['a'],
                    filt_samples, axis=0, zi=self.lp_filt['zi'])
            if self.debug_outputs:
                self._send_output_vec(smooth_eeg_samples, timestamp, 'smooth_eeg')
        else:
            smooth_eeg_samples = filt_samples
        self.smooth_eeg_buffer.update(smooth_eeg_samples)

        if config['filter_bank']:
            filter_bank_samples = {}
            for name, filt_dict in self.filter_bank.items():
                filter_bank_samples[name], self.filter_bank[name]['zi'] = \
                    signal.lfilter(filt_dict['b'], filt_dict['a'],
                                   filt_samples, axis=0,
                                   zi=self.filter_bank[name]['zi'])
            low_freq_chs = filter_bank_samples['delta'][0, [0, 2]] #+ filter_bank_samples['theta'][0, [0, 1]

        window = self.smooth_eeg_buffer.extract(self.window_len)

        eegEarWindow = window[:, 3] #data from right ear Channel
        #eye movement computed from the difference between two frontal channels
        eyewindow = self.smooth_eeg_buffer.extract(200)
        eegFLWindow = eyewindow[:, 1]
        eegFRWindow = eyewindow[:, 2]
#        norm_diff_eyes = eegFLWindow[-1] - eegFRWindow[-1]*np.nanstd(eegFLWindow, axis=0)/np.nanstd(eegFRWindow, axis=0)
#        eyeH = np.reshape([np.square(norm_diff_eyes)], (1, 1))

        #find blinks in the left eegEarWindow
        blinkVal = ut.blink_template_match(eegEarWindow)
        if (blinkVal > 100000 and self.blink == 0):
            self.blink = 50
            self.blinkwait = 350
        else:
            if (self.blinkwait > 0):
                self.blinkwait -= 1
            if (self.blink > 0):
                self.blink -= 1

        # LONGER-TERM CALM SCORE based on Saccadic Eye Movement
        eye_mov_percent = np.reshape(np.percentile(eegFLWindow - eegFRWindow, 90), (1, 1))
        self.eye_mov_percent_buffer.update(eye_mov_percent)
        remap_eye_mov_percent = ut.sigmoid(self.eye_mov_percent_buffer.extract().mean(), 0.5, -10, 0)


        max_value = 1
        incr_decr = remap_eye_mov_percent < 0.2
        inc = self.increments_buffer.extract().mean()
        dpoints_per_second = 0.0005

        if incr_decr:
            self.slow_calm_score += dpoints_per_second*inc # 1/max([max_value - self.slow_calm_score, 1])
        else:
            self.slow_calm_score -= dpoints_per_second*inc*4 #0.7 # (self.slow_calm_score)/1280


        self.increments_buffer.update(np.reshape(incr_decr, (1, 1)))

        if self.slow_calm_score > max_value:
            self.slow_calm_score = max_value
        elif self.slow_calm_score < 0:
            self.slow_calm_score = 0

        self.slow_calm_score_buffer.update(np.reshape(self.slow_calm_score, (1, 1)))


       # Send outputs at a reduced sampling rate
        if self.smooth_eeg_buffer.pts%3==0 :
            self._send_output_vec(smooth_eeg_samples, timestamp, 'muse/eeg')
            if (self.blink > 0):
                self._send_output(np.array([[1]]), timestamp, 'blink')
            else:
                self._send_output(np.array([[0]]), timestamp, 'blink')
            self._send_output(blinkVal/300000,timestamp,'blinkVal')
            self._send_output(remap_eye_mov_percent, timestamp, 'saccad')

            self._send_output(np.reshape(self.slow_calm_score_buffer.extract().mean(), (1, 1)),timestamp, 'calm') # slow_calm_score
            self._send_output(low_freq_chs / self.low_freq_chs_std + 0.5, timestamp, 'low_freq_chs')

        # process and send output at every step.   usually about every 1/10s
        if self.eeg_buffer.pts > self.step:
            self.eeg_buffer.pts = 0

            # Get filtered EEG window
            if config['lpfilter']:
                window = self.smooth_eeg_buffer.extract(self.window_len)
            else:
                window = self.eeg_buffer.extract(self.window_len)
            psd_raw_buffer = self.eeg_buffer.extract(self.window_len)

            # Get average PSD
            psd, f = ut.fft_continuous(psd_raw_buffer, n=int(self.fs), psd=True,
                                       log='psd', fs=self.fs, window='hamming')
            self.psd_buffer.update(np.expand_dims(psd, axis=0))
            mean_psd = np.nanmean(self.psd_buffer.extract(), axis=0)

            # find variance of eegWindow  for Bad Signal detact
            eegVar = np.nanvar(window,axis=0)
            self._send_output_vec(eegVar.reshape(1,self.n_channels), timestamp, 'hsi')

            if (self.sparseOutput!=None):
                 #send channel varience for signal quality indication at source Raspberry Pi
                #send(Address('10.0.0.14','1234'), "/hsi", eegVar[0],eegVar[1],eegVar[2],eegVar[3])
                self._send_sparseOutput_vec(eegVar.reshape(1,self.n_channels), timestamp, 'hsi')


            # Get band powers and ratios

            bandPowers, bandNames = ut.compute_band_powers(mean_psd, f, relative=False)
            ratioPowers, ratioNames = ut.compute_band_ratios(bandPowers)

            if  (self.firstWindowProc):
                self.band_powers = bandPowers
                self.band_names = bandNames
                self.ratio_powers = ratioPowers
                self.ratio_names = ratioNames
                self.scores = np.zeros((len(self.band_names), self.n_channels))
                self.firstWindowProc = False

            if (eegVar.mean() < 300 and self.blinkwait == 0 ):  #threshold for good data
                for i, (name, hist) in enumerate(self.hists.items()):
                    self.band_powers = bandPowers
                    self.ratio_powers = ratioPowers
                #send good data indicator based on mean eegWindow variance and blinkwait
                    self._send_output(np.array([[1]]), timestamp, 'goodData') #good data
            else:
                self._send_output(np.array([[0]]), timestamp, 'goodData') #good data

            self._send_outputs(self.band_powers, timestamp, 'bands')
            self._send_outputs(self.ratio_powers, timestamp, 'ratios')


            mask = ((f >= 30 ) & (f<50))

            self.low_freq_chs_buffer.update(np.reshape(low_freq_chs, (1, -1)))
            self.low_freq_chs_std = self.low_freq_chs_buffer.extract().std(axis=0)

            emg_power = np.mean(mean_psd[mask, 0], axis=0) #HF power of right ear
            self._send_output(np.array([np.sqrt(emg_power)/2]), timestamp, 'emg')

    def _process_acc(self, samples, timestamp):
        self._send_output_vec(samples,0,'muse/acc')

        self.acc_buffer.update(samples)
        window = self.acc_buffer.extract(self.acc_window_len)

        timestamps = np.linspace(0,1/50*self.acc_window_len,self.acc_window_len)
        new_fs= 250
        timestamps_upsampled = np.arange(timestamps[0], timestamps[-1],1/new_fs)
        f = interpolate.interp1d(timestamps, window, kind='cubic', axis=0,
                    fill_value=np.nan, assume_sorted=True)
        window_upsampled = f(timestamps_upsampled)
        for t in range(timestamps_upsampled.size-5,timestamps_upsampled.size):
            if self.debug_outputs:
                self._send_output(window_upsampled[t],0,'upsamp')
            upsample = np.array(window_upsampled[t]).reshape(1,3)
            filt_samples, self.filter0['zi'] = signal.lfilter(
                   self.filter0['b'], self.filter0['a'],
                   upsample, axis=0, zi=self.filter0['zi'])
            self.filt0_buffer.update(filt_samples)
            if self.debug_outputs:
                self._send_outputs(filt_samples,0,'filter0')

            filt_samples, self.filter1['zi'] = signal.lfilter(
                   self.filter1['b'], self.filter1['a'],
                   filt_samples, axis=0, zi=self.filter1['zi'])
            if self.debug_outputs:
                self._send_outputs(filt_samples,0,'filter1')

            filt_samples = np.sqrt(np.sum(filt_samples ** 2, axis=1))
            if self.debug_outputs:
                self._send_output(filt_samples,0,'filter1L2')

            heart_samples, self.filter2['zi'] = signal.lfilter(
                    self.filter2['b'], self.filter2['a'],
                    filt_samples, axis=0, zi=self.filter2['zi'])

            if self.debug_outputs:
                self._send_output(heart_samples,0,'filter2')

            breathfilt_samples, self.filter3['zi'] = signal.lfilter(
                    self.filter3['b'], self.filter3['a'],
                    upsample, axis=0, zi=self.filter3['zi'])
            if self.debug_outputs:
                self._send_outputs(breathfilt_samples,0,'filter3')
        self.heart_buffer.update(heart_samples.reshape(1,1))
        heartbuf = self.heart_buffer.extract(150)
        heartbufMin = heartbuf.min()
        heartbufMax = heartbuf.max()
        heart = np.reshape((heartbuf[-1]-heartbufMin)/(heartbufMax-heartbufMin), (1, 1))
        self._send_output(heart,0,'heart')


        breathSmooth = breathfilt_samples[0,2].reshape(1,)
        if self.debug_outputs:
            self._send_output(breathSmooth,0,'breathRaw')

        breathSmooth, self.filter4['zi'] = signal.lfilter(
            self.filter4['b'], self.filter4['a'],
            breathSmooth, axis=0, zi=self.filter4['zi'])
        if self.debug_outputs:
            self._send_output(breathSmooth,0,'breathSmooth')

        breathNorm, self.filter5['zi'] = signal.lfilter(
            self.filter5['b'], self.filter5['a'],
            breathSmooth, axis=0, zi=self.filter5['zi'])

        if self.debug_outputs:
            self._send_output(breathNorm,0,'breathNorm')

        breathFast, self.filter6['zi'] = signal.lfilter(
            self.filter6['b'], self.filter6['a'],
            breathSmooth, axis=0, zi=self.filter6['zi'])

        if self.debug_outputs:
            self._send_output(breathFast,0,'breathFast')

        breathLow, self.filter7['zi'] = signal.lfilter(
			self.filter7['b'], self.filter7['a'],
			breathSmooth, axis=0, zi=self.filter7['zi'])
        if self.debug_outputs:
            self._send_output(breathLow,0,'breathLow')


        self.breath_buffer.update(breathLow.reshape(1,1))
        breathbuf = self.breath_buffer.extract(1000)
        breathbufMin = breathbuf.min()
        breathbufMax = breathbuf.max()
        breath = np.reshape((breathbuf[-1]-breathbufMin)/(breathbufMax-breathbufMin), (1, 1))
        self._send_output(breath,0,'breath')

    def _process_gyro(self, samples, timestamp):
        self._send_output_vec(samples,0,'muse/gyro')


    def _send_outputs(self, output, timestamp, name):
        """Send pipeline outputs through the LSL or OSC stream.

        Args:
            output (scalar): output of the pipeline
            timestamp (float): timestamp
        """
        for out in self._output_threads:
            if isinstance(out, str):  # LSL outlet
                self._outlet.push_sample([output], timestamp=timestamp)

            else:  # OSC output stream
                if USE_LIBLO:
                    for c in range(self.n_channels):
                        new_output = [('f', x) for x in output[:, c]]
                        message = Message('/{}{}'.format(name, c), *new_output)
                        #send(out, Bundle(timestamp, message))
                        send(out,  message)
                else:
                    for c in range(self.n_channels):
                        self._client.send_message('/{}{}'.format(name, c),
                                                  output[:, c])

            if self.verbose:
                print('Output: {}'.format(output))
    def _send_output_vec(self, output, timestamp, name):
        """Send pipeline outputs through the LSL or OSC stream.

        Args:
            output (scalar): output of the pipeline
            timestamp (float): timestamp
        """
        for out in self._output_threads:
            if isinstance(out, str):  # LSL outlet
                self._outlet.push_sample([output], timestamp=timestamp)

            else:  # OSC output stream
                if USE_LIBLO:
                    new_output = [('f', x) for x in output[0,:]]
                    message = Message('/{}'.format(name), *new_output)
#                    send(out, Bundle(timestamp, message))
                    send(out, message)
            if self.verbose:
                print('Output: {}'.format(output))

    def _send_sparseOutput_vec(self, output, timestamp, name):
        """Send pipeline outputs through the LSL or OSC stream.

        Args:
            output (scalar): output of the pipeline
            timestamp (float): timestamp
        """
        for out in self._sparseOutput_threads:
            if isinstance(out, str):  # LSL outlet
                self._outlet.push_sample([output], timestamp=timestamp)

            else:  # OSC output stream
                if USE_LIBLO:
                    new_output = [('f', x) for x in output[0,:]]
                    message = Message('/{}'.format(name), *new_output)
                    #send(out, Bundle(timestamp, message))
                    send(out, message)
            if self.verbose:
                print('sparseOutput: {}'.format(output))


    def _send_output(self, output, timestamp, name):
        """Send pipeline outputs through the LSL or OSC stream.
        NOT PER CHANNEL
        Args:
            output (scalar): output of the pipeline
            timestamp (float): timestamp
        """
        for out in self._output_threads:
            if isinstance(out, str):  # LSL outlet
                raise NotImplementedError
                # self._outlet.push_sample([output], timestamp=timestamp)

            else:  # OSC output stream
                if USE_LIBLO:
                    if (np.array(output).size==1):
                         new_output = [('f', np.asscalar(output))]
                         message = Message('/{}'.format(name), *new_output)
                    else:
                         new_output = [('f', x) for x in output[:]]
                         message = Message('/{}'.format(name), *new_output)
 #                   send(out, Bundle(timestamp, message))
                    send(out, message)
                else:
                    raise NotImplementedError
#                    self._client.send_message(}{}'.format(name),output[:])
            if self.verbose:
                print('Output: {}'.format(output))

    def _send_sparseOutput(self, output, timestamp, name):
        for out in self._sparseOutput_threads:
            if isinstance(out, str):  # LSL outlet
                raise NotImplementedError
            else:  # OSC output stream
                if USE_LIBLO:
                    if (np.array(output).size==1):
                         new_output = [('f', np.asscalar(output))]
                         message = Message('/{}'.format(name), *new_output)
                    else:
                         new_output = [('f', x) for x in output[:]]
                         message = Message('/{}'.format(name), *new_output)
                    #send(out, Bundle(timestamp, message))
                    send(out, message)
                else:
                    raise NotImplementedError
            if self.verbose:
                print('spareOutput: {}'.format(output))

    def start(self):
        """Start receiving and processing EEG data.
        """
        self.started = True

        if isinstance(self.incoming, str):  # LSL inlet
            self.eeg_thread = Thread(target=self._update_eeg_lsl)
            self.eeg_thread.daemon = True
            self.eeg_thread.start()
        else:  # OSC input stream
            if USE_LIBLO:
                self._osc_server.add_method('/muse/eeg', None,
                                            self._update_eeg_liblo_osc)
                self._osc_server.add_method('/muse/acc', None,
                                            self._update_acc_liblo_osc)
                self._osc_server.add_method('/muse/gyro', None,
                                            self._update_gyro_liblo_osc)
                self._osc_server.start()
            else:
                self._dispatcher.map('Person2/eeg',
                                     self._update_eeg_python_osc, 'EEG')
                self._osc_server = osc_server.ThreadingOSCUDPServer(
                        ('127.0.0.1', self.incoming['port']), self._dispatcher)
                print('OSC server initialized at port {}.'.format(
                        self.incoming['port']))
                self._server_thread = Thread(
                        target=self._osc_server.serve_forever)
                self._server_thread.start()

    def stop(self):
        """
        """
        self.started = False

        if isinstance(self.incoming, dict):
            if not USE_LIBLO:
                self._server_thread.shutdown()


if __name__ == '__main__':


    #EEG PROCESSING
    FS = 256.
    #AC notch filter
    EEG_b, EEG_a = ut.get_filter_coeff(FS, 6, l_freq=65, h_freq=55,method='butter')
    #demean
    EEG_b2, EEG_a2 = ut.get_filter_coeff(FS, 3, l_freq=1,method='butter')
    #eeg range for clean signal display
    EEG_b3, EEG_a3 = ut.get_filter_coeff(FS, 3, h_freq=40,method='butter')
    # Filter bank
    b_delta, a_delta = ut.get_filter_coeff(FS, 3, l_freq=1, h_freq=4, method='butter')
    b_theta, a_theta = ut.get_filter_coeff(FS, 3, l_freq=4, h_freq=7.5, method='butter')
    b_alpha, a_alpha = ut.get_filter_coeff(FS, 3, l_freq=7.5, h_freq=13, method='butter')
    b_beta, a_beta = ut.get_filter_coeff(FS, 3, l_freq=13, h_freq=30, method='butter')


    #Motion Sensor Processing
    FSb1 = 250.
    b0 = np.array([-1, 2, -1]) / 3
    a0 = 1
    b1, a1 = ut.get_filter_coeff(FSb1, 4, l_freq=10, h_freq=13,method='butter')
    b2, a2 = ut.get_filter_coeff(FSb1, 2, l_freq=0.75, h_freq=2.5,method='butter')
    min_breath_period = 0.3  # maximal breath frequency
    min_n_points_in_breath = int(min_breath_period * FSb1)
    b3 = np.ones((min_n_points_in_breath,)) / min_n_points_in_breath
    a3 = 1
    FSb2 = 50.
    b4, a4 = ut.get_filter_coeff(FSb2, 3, h_freq=5,method='butter')
    b5, a5 = ut.get_filter_coeff(FSb2, 3, l_freq=0.13, h_freq=1,method='butter')
    b6, a6 = ut.get_filter_coeff(FSb2, 3, l_freq=1, h_freq=5,method='butter')
    b7, a7 = ut.get_filter_coeff(FSb2, 3, h_freq=1,method='butter')

    config = {'fs': FS,
              'n_channels': 5,
              'raw_buffer_len': int(3 * FS),
              'filt_buffer_len': int(3 * FS),
              'window_len': int(FS),
              'step': int(FS / 10),
              'filter': (EEG_b, EEG_a),
              'hpfilter': (EEG_b2, EEG_a2),
              'lpfilter': (EEG_b3, EEG_a3),
              'filter_bank': {'delta': (b_delta, a_delta)}, #'theta': (b_theta, a_theta)}, #'alpha': (b_alpha, a_alpha), 'beta': (b_beta, a_alpha)},
              'psd_window_len': int(FS),
              'psd_buffer_len': 5,
              'filter0': (b0, a0),
              'filter1': (b1, a1),
              'filter2': (b2, a2),
              'filter3': (b3, a3),
              'filter4': (b4, a4),
              'filter5': (b5, a5),
              'filter6': (b6, a6),
              'filter7': (b7, a7),
              }

    fft_server = FFTServer({'port':options.port},  # 'MEG', #  #
                            ({'address': options.oscip, 'port': options.oscport}),
                            ({'address': options.sparseip, 'port': options.sparseport}),
                           config=config,
                           device_source='muse',  #vive,  leroy
                           software_source='muselsl',
                           debug_outputs=False,
                           verbose=False)
    fft_server.start()

    while True:
        try:
            time.sleep(1)
        except:
            fft_server.stop()
            print('breaking')
            break


