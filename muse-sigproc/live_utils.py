#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for real-time processing
"""

from collections import OrderedDict
from threading import Thread, Event

import numpy as np
from scipy import signal


BAND_FREQS = OrderedDict()
BAND_FREQS['delta'] = (1, 4)
BAND_FREQS['theta'] = (4, 8)
BAND_FREQS['alpha'] = (7.5, 13)
BAND_FREQS['beta'] = (13, 30)
BAND_FREQS['gamma'] = (30, 44)

RATIOS = OrderedDict()
RATIOS['beta/alpha'] = (3, 2)
RATIOS['theta/alpha'] = (4, 2)

BLINKWAVE = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        -17.683,-31.44,-44.534,-56.351,-66.451,-74.543,-80.462,
                        -84.142,-85.604,-84.934,-82.271,-77.79,-71.697,-64.214,
                        -55.574,-46.011,-35.757,-25.037,-14.061,-3.028,7.882,
                        18.507,28.704,38.352,47.346,55.605,63.064,69.678,
                        75.418,80.272,84.239,87.336,89.588,91.031,91.709,
                        91.673,90.981,89.691,87.868,85.577,82.882,79.848,
                        76.538,73.011,69.325,65.534,61.686,57.826,53.994,
                        50.223,46.544,42.981,39.552,36.273,33.154,30.198,
                        27.408,24.782,22.313,19.993,17.813,15.758,13.815,
                        11.969, 10.204,8.5049,6.8558,5.2423,3.6505,2.0681])


HEARTWAVE = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21.623,36.971,59.776,98.036,132.82,175.89,185.95,115.3,-24.952,-202.51,-364.39,-450.96,-412.91,-324.91,-304.24,-333.27,-347.46,-342.08,-261.02,-81.866,84.638,190.3,250.39,279.78,281.17,266.42,256.18,254.9,245.22,229.63,217.99,210.21,197.21,177.98,168.21,161.73,145.03,131.1,121.96,110.82,101.6,88.15,86.381,86.017,72.336,63.964,61.159,51.928,43.105,44.441,42.399,32.123])


def sigmoid(x, a=1, b=0, c=0):
    """Sigmoid function.

    Args:
        x (array_like): values to map
        a (float): control the steepness of the curve
        b (float): control the shift in x
        c (float): control the shift in y

    Returns:
        (numpy.ndarray) output of the sigmoid
    """
    return 1 / (1 + np.exp(-(a*x + b))) + c


def get_filter_coeff(fs, N, l_freq=None, h_freq=None, method='butter'):
    """Get filter coefficients.

    Args:
        fs (float): sampling rate of the signal to filter
        N (int): order of the filter

    Keyword Args:
        l_freq (float or None): lower cutoff frequency in Hz. If provided
            without `h_freq`, returns a highpass filter. If both `l_freq`
            and `h_freq` are provided and `h_freq` is larger than `l_freq`,
            returns a bandpass filter, otherwise returns a bandstop filter.
        h_freq (float or None): higher cutoff frequency in Hz. If provided
            without `l_freq`, returns a lowpass filter.
        method (string): method to compute the coefficients ('butter', etc.)

    Returns:
        (numpy.ndarray): b coefficients of the filter
        (numpy.ndarray): a coefficients of the filter

    Examples:
        Get a 5th order lowpass filter at 30 Hz for a signal sampled at 256 Hz
        >>> b, a = get_filter_coeff(256, 5, h_freq=30)
    """

    if l_freq is not None and h_freq is not None:
        if l_freq < h_freq:
            btype = 'bandpass'
            Wn = [l_freq/(float(fs)/2), h_freq/(float(fs)/2)]
        elif l_freq > h_freq:
            btype = 'bandstop'
            Wn = [h_freq/(float(fs)/2), l_freq/(float(fs)/2)]
    elif l_freq is not None:
        Wn = l_freq/(float(fs)/2)
        btype = 'highpass'
    elif h_freq is not None:
        Wn = h_freq/(float(fs)/2)
        btype = 'lowpass'

    if method == 'butter':
        b, a = signal.butter(N, Wn, btype=btype)
    else:
        raise(ValueError('Method ''{}'' not supported.'.format(method)))

    return b, a


# def plot_freq_response(b, a, fs):
#     """Plot the frequency response of a filter.

#     Args:
#         b (numpy.ndarray): coefficients `b` of the filter
#         a (numpy.ndarray): coefficients `a` of the filter
#         fs (float): sampling frequency of the signal to be filtered

#     Returns:
#         (matplotlib.figure.Figure) : figure
#         (matplotlib.axes.Axes) : axes of the plot

#     Taken from https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/
#         scipy.signal.freqz.html
#     """
#     w, h = signal.freqz(b, a)
#     f = w/np.pi*fs/2

#     fig, ax = plt.subplots()
#     ax.set_title('Digital filter frequency response')
#     ax.plot(f, 20 * np.log10(abs(h)), 'b')
#     ax.set_ylabel('Amplitude [dB]', color='b')
#     ax.set_xlabel('Frequency [Hz]]')

#     ax2 = ax.twinx()
#     angles = np.unwrap(np.angle(h))
#     ax2.plot(f, angles, 'g')
#     ax2.set_ylabel('Angle (radians)', color='g')

#     plt.grid()
#     plt.axis('tight')
#     plt.show()

#     return fig, ax



def blink_template_match(eegEar):
    blinkVal = (np.reshape(BLINKWAVE,(1,256))*np.reshape(eegEar,(1,256))).sum()
    return blinkVal

def heart_template_match(window):
    ecgwindow = window[window.size-HEARTWAVE.size:window.size]
    #heartVal = (np.reshape(HEARTWAVE,(1,HEARTWAVE.size)*np.reshape(ecgwindow,(1,HEARTWAVE.size))).sum()
    heartVal = np.dot(HEARTWAVE.T,ecgwindow).sum()
    return heartVal

def fft_continuous(data, n=None, psd=False, log='log', fs=None,
                   window='hamming'):
    """Apply the Fast Fourier Transform on continuous data.

    Apply the Fast Fourier Transform algorithm on continuous data to get
    the spectrum.
    Steps:
        1- Demeaning
        2- Apply hamming window
        3- Compute FFT
        4- Grab lower half

    Args:
        data (numpy.ndarray): shape (`n_samples`, `n_channels`). Data for
            which to get the FFT

    Keyword Args:
        n (int): length of the FFT. If longer than `n_samples`, zero-padding
            is used; if smaller, then the signal is cropped. If None, use
            the same number as the number of samples
        psd (bool): if True, return the Power Spectral Density
        log (string): can be 'log' (log10(x)), 'log+1' (log10(x+1)) or None
        fs (float): Sampling rate of `data`.
        window (string): if 'no_window' do not use a window before
            applying the FFT. Otherwise, use as the window function.
            Currently only supports 'hamming'.

    Returns:
        (numpy.ndarray) Fourier Transform of the original signal
        (numpy.ndarray): array of frequency bins
    """
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    [n_samples, n_channels] = data.shape

    data = data - data.mean(axis=0)
    if window.lower() == 'hamming':
        H = np.hamming(n_samples).reshape((-1, 1))
    elif window.lower() == 'no_window':
        H = np.ones(n_samples).reshape((-1, 1))
    else:
        raise ValueError('window value {} is not supported'.format(window))
    L = np.min([n_samples, n]) if n else n_samples
    Y = np.fft.fft(data * H, n, axis=0) / L
    freq_bins = (fs * np.arange(0, Y.shape[0] / 2 + 1) / Y.shape[0]) \
        if fs is not None else None

    out = Y[0:int(Y.shape[0] / 2) + 1, :]
    out[:, 0] = 2 * out[:, 0]

    if psd:
        out = np.abs(out) ** 2
    if log == 'log':
        out = np.log10(out)
    elif log == 'log+1':
        out = np.log10(out + 1)

    return out, freq_bins


def compute_band_powers(psd, f, relative=False, band_freqs=BAND_FREQS):
    """Compute the standard band powers from a PSD.

    Compute the standard band powers from a PSD.

    Args:
        psd (numpy.ndarray): array of shape (n_freq_bins, n_channels)
            containing the PSD of each channel
        f (array_like): array of shape (n_freq_bins,) containing the
            frequency of each bin in `psd`

    Keyword Args:
        relative (bool): if True, compute relative band powers
        band_freqs (OrderedDict): dictionary containing the band names as
            keys, and tuples of frequency boundaries as values. See
            BAND_FREQS.

    Returns:
        (numpy.ndarray): array of shape (n_bands, n_channels) containing
            the band powers
        (list): band names
    """
    band_powers = np.zeros((len(band_freqs), psd.shape[1]))
    for i, bounds in enumerate(band_freqs.values()):
        mask = (f >= bounds[0]) & (f <= bounds[1])
        band_powers[i, :] = np.mean(psd[mask, :], axis=0)

    if relative:
        band_powers /= band_powers.sum(axis=0)

    return band_powers, list(band_freqs.keys())


def compute_band_ratios(band_powers, ratios=RATIOS):
    """Compute ratios of band powers.

    Args:
        band_powers (numpy.ndarray): array of shape (n_bands, n_channels)
            containing the band powers

    Keyword Args:
        ratios (tuple of tuples): contains the indices of band powers to
            compute ratios from. E.g., ((3, 2)) is beta/alpha.
            See BAND_FREQS and RATIOS.
        ratios (OrderedDict): dictionary containing the ratio names as keys
            and tuple of indices of the bands to used for each ratio. See
            RATIOS.

    Returns:
        (numpy.ndarray): array of shape (n_rations, n_channels)
            containing the ratios of band powers
        (list): ratio names
    """
    ratio_powers = np.zeros((len(ratios), band_powers.shape[1]))
    for i, ratio in enumerate(ratios.values()):
        ratio_powers[i, :] = band_powers[ratio[0], :] / band_powers[ratio[1], :]

    return ratio_powers, list(ratios.keys())


class CircularBuffer(object):
    """Circular buffer for multi-channel 1D or 2D signals

    Circular buffer for multi-channel 1D or 2D signals (could be increased
    to arbitrary number of dimensions easily).

    Attributes:
        buffer (numpy.ndarray): array (n_samples, n_channels(, n_points))
            containing the data
        noise (numpy.ndarray): array (n_samples, n_channels(, n_points)) of
            booleans marking bad data
        ind (int): current index in the circular buffer
        pts (int): total number of points seen so far in the buffer
        n (int): length of buffer (number of samples)
        m (int): number of channels

    Args:
        n (int): length of buffer (number of samples)
        m (int): number of channels

    Keyword Args:
        p (int): (optional) length of third dimension for 3D buffer
        fill_value (float): value to fill the buffer with when initializing

    Note:
        The indexing syntax of numpy lets us extract data from all trailing
        dimensions (e.g. x[0, 2] = x[0, 2, :]). This makes it easy to add
        dimensions.

    Todo:
        - Implement 3D buffering with argument `p`.
        - Add `pts`, `noise` and `buffer` as properties?
    """

    def __init__(self, n, m, p=None, fill_value=0.):
        self.n = int(n)
        self.m = int(m)

        if p:
            self.p = int(p)
            self.buffer = np.zeros((self.n, self.m, self.p)) + fill_value
            self.noise = np.zeros((self.n, self.m, self.p), dtype=bool)
        else:
            self.buffer = np.zeros((self.n, self.m)) + fill_value
            self.noise = np.zeros((self.n, self.m), dtype=bool)

        self.ind = 0
        self.pts = 0

    def update(self, x):
        """Update the buffer.

        Args:
            x (numpy.ndarray): array of shape
                (n_new_samples, n_channels(, n_points))
        """
        if x.ndim != self.buffer.ndim:
            raise ValueError('x has not the same number of dimensions as '
                             'the buffer.')
        nw = x.shape[0]

        # Determine index at which new values should be put into array
        ind = np.arange(self.ind, self.ind + nw, dtype=np.int16) % self.n
        self.buffer[ind, :] = x

        # Set self.ind = to the index at which new locations were put.
        # Separately defined here to allow new data to be an array rather
        # than just one row
        self.ind = (ind[-1] + 1) % self.n
        self.pts += nw

    def extract(self, nw=None):
        """Extract sample(s) from the buffer.

        Keyword Args:
            nw (int): number of samples to extract from the buffer. If
                None, return n points.
        """
        if not nw:
            nw = self.n

        ind = np.arange(self.ind - nw, self.ind, dtype=np.int16) % self.n
        return self.buffer[ind, :]

    def mark_noise(self, noise, nw=None):
        """Mark noisy samples in the buffer.

        Mark the last `nw` samples in the buffer as noisy (noisy -> True;
        clean -> False).

        Args:
            noise (bool): if True, mark the last nw samples as noise

        Keyword Args:
            nw (int): number of samples to mark as noise. If None, use n
                points.
        """
        if not nw:
            nw = self.n

        ind = np.arange(self.ind - nw, self.ind, dtype=np.int16) % self.n
        self.noise[ind, :] = noise

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, int) or value < 1:
            raise TypeError('n must be a non-zero positive integer.')
        self._n = value

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        if not isinstance(value, int) or value < 1:
            raise TypeError('m must be a non-zero positive integer.')
        self._m = value


class NanBuffer(CircularBuffer):
    """Circular buffer that can accomodate missing values (NaNs).

    Circular buffer that can accomodate missing values coming in as NaNs.
    Previous values are repeated as long as NaNs are received. When a new
    valid value finally is received, linear interpolation is performed to
    replace the chunk of missing values.

    Attributes:
        nan_buffer (numpy.ndarray): array with the same shape as the data
            buffer, but containing only 0, 1 or 2.
            0 -> no NaN
            1 -> NaN
            2 -> interpolated NaNs
        nan_start_ind (numpy.ndarray): indices of the start of a NaN streak
            with shape (n_channels, )
        prev_ind (int): index at which the previous sample was put in the
            buffer

    Args:

    """
    def __init__(self, n, m, p=None, fill_value=0.):
        """
        """
        if p:
            raise NotImplementedError('3D arrays are not yet supported.')

        super().__init__(n, m, p=p, fill_value=fill_value)

        self.nan_buffer = np.zeros_like(self.buffer, dtype=np.int8)
        self.nan_start_ind = np.zeros((self.m,), dtype=np.int8) - 1
        self.prev_ind = self.n - 1

    def update(self, x):
        """Update the buffer.

        Args:
            x (numpy.ndarray): array of shape
                (n_new_samples, n_channels(, n_points))
        """
        if x.ndim != self.buffer.ndim:
            raise ValueError('x has not the same number of dimensions as '
                             'the buffer.')
        if x.shape[0] > 1:
            raise NotImplementedError('Updating with more than one sample '
                                      'at once is not supported yet.')

        # Determine index at which new values should be put into array
        ind = [x % self.n for x in range(self.ind, self.ind + 1)]

        # Manage NaNs
        self.nan_buffer[ind, :] = np.logical_or((x == 0), np.isnan(x))
        for c in range(self.m):
            # Case where the current sample is a NaN
            if self.nan_buffer[ind, c] == 1:
                # First case: NaNs at the very end of the buffer -> Set to
                # previous good value
                x[0, c] = self.buffer[self.prev_ind, c]

                if self.nan_start_ind[c] == -1:
                    self.nan_start_ind[c] = self.prev_ind

            # Case where the previous sample was a NaN, but the current
            # sample is good (NaN streak in the middle of the buffer)
            elif (self.nan_buffer[ind, c] == 0) and (self.nan_buffer[self.prev_ind, c] == 1):

                # Find the boundaries of the NaN streak
                n_cont_nans = np.mod(self.n + (self.ind - self.nan_start_ind[c]), self.n) - 1

                if n_cont_nans > self.n:
                    # If we reach a point where we have more continuous
                    # NaNs than the size of the window, change the
                    # nanStartInd to the start of that window. This is done
                    # to avoid calling returnInds with a number of
                    # continuous NaNs higher than approximately 300.
                    self.nan_start_ind[c] = np.mod(self.n + (self.ind - self.n), self.n)
                    n_cont_nans = self.n

                # Find the indices of values to replace
                indices = [x % self.n for x in range(self.ind - n_cont_nans, self.ind)]

                # Linearly interpolate
                intercept = self.buffer[self.nan_start_ind[c], c]
                slope = (x[0, c] - intercept) / (n_cont_nans + 1)
                self.buffer[indices, c] = slope * np.arange(1, n_cont_nans + 1) + intercept

                # Set this streak of NaNs to 2 in the nan buffer (to
                # distinguish them from the other cases when computing the
                # FFT)
                self.nan_buffer[indices, c] = 2

                # Reset nan_start_ind to -1 because the NaN streak is over
                self.nan_start_ind[c] = -1

        self.buffer[ind, :] = x
        self.prev_ind = self.ind
        self.ind = (ind[-1] + 1) % self.n
        self.pts += 1


class Histogram(object):
    """Fixed-size histogram for live-scoring a multi-channel random variable.

    Attributes:
        hist (numpy.ndarray): shape (n_bins, n_channels), containing counts
            for each histogram bins
        cum_hist (numpy.ndarray): shape (n_bins, n_channels), containing
            cumulative sums for each histogram bins. This is useful to
            compute percentiles.
        bins (numpy.ndarray): shape (n_bins + 1,), containing the boundary
            values of each of the `n_bins` bins
        counts (int): number of values collected up until now
        min_count (int): minimum count before percentiles can be computed
        decay (float): value between 0 and 1 which controls the relative
            importance of new samples when updating the histogram. A value
            of 0 means that old values are ignored and only new values are
            used; a value of 1 means that old values never fade, and that
            new values are simply added to the existing histogram.

    Args:
        n_bins (int): number of bins in the histogram
        n_channels (int): number of channels

    Keyword Args:
        bounds (tuple): (min_value, max_value) to collect in the histogram
        min_count (int): minimum count before percentiles can be computed
        decay (float): see Attributes description.
    """
    def __init__(self, n_bins, n_channels, bounds=(-2, 8), min_count=0,
                 decay=1):
        self.n_bins = n_bins
        self.bins = np.linspace(bounds[0], bounds[1], n_bins + 1)
        self.n_channels = n_channels
        self.hist = np.zeros((n_bins, n_channels))
        self.cum_hist = np.zeros((n_bins, n_channels))
        self.counts = 0
        self.min_count = min_count
        self.decay = decay

    def get_prct_and_add(self, x):
        """Get the percentile of a new point, then add it to the histogram.

        Args:
            x (array_like): points to add to the histogram, of shape
                (n_channels, )

        Returns:
            (numpy.ndarray): percentiles of `x`
        """
        if self.decay != 1:
            self.hist *= self.decay

        prcts = np.zeros((self.n_channels, ))
        for c in range(self.n_channels):
            # Find corresponding bin
            ind = self._find_bin_ind(x[c])
            if self.counts > self.min_count:
                prcts[c] = self.cum_hist[ind, c]/self.cum_hist[self.cum_hist.shape[0]-1,c]
                 #               prcts[c] = self.cum_hist[ind, c] / self.counts
            self.hist[ind, c] += 1

        # Update cumulative sum
        self.cum_hist = self.hist.cumsum(axis=0)
        self.counts += 1

        return prcts

    def reset(self):
        """Reset the histogram and cumulative sum.
        """
        self.bins *= 0
        self.cum_hist *= 0
        self.counts = 0

    def _find_bin_ind(self, x):
        """Find histogram bin index.

        Args:
            x (float): value for which to find the bin index

        Returns:
            (int): bin index
        """
        inds = np.where(self.bins >= x)[0].tolist()
        return self.n_bins - 1 if not inds else inds[0] - 1


class Timer(Thread):
    """Repeating timer object.

    This timer calls its callback function every `interval` seconds, until
    its `stop` method is called.

    Args:
        interval (float): interval between callbacks (in seconds)
        callback (function): function to call every `interval` seconds

    """
    def __init__(self, interval, callback):
        Thread.__init__(self)
        self.interval = interval
        self.callback = callback
        self.stopped = Event()

    def run(self):
        while not self.stopped.wait(self.interval):
            self.callback()

    def stop(self):
        self.stopped.set()
