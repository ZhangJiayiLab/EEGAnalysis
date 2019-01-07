import scipy.signal as signal
import numpy as np

def _butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, bandrange, fs, order=4):
    b, a = signal.butter(order, [bandrange[0]/fs*2, bandrange[1]/fs*2], 'bandpass')
    y = signal.filtfilt(b, a, data)
    return y
    
def _bessel_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.bessel(order, normal_cutoff, btype='highpass')
    return b, a


def bessel_highpass_filter(data, cutoff, fs, order=5):
    b, a = _bessel_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def gaussian_kernel(fs, sigma):
    ktime = np.linspace(-1, 1, int(2 * fs))
    kernel = 1 / np.sqrt(2*np.pi*sigma**2) * np.exp(-ktime ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def gaussianwind(data, fs, sigma):
    k = gaussian_kernel(fs, sigma)
    totalpwr_filter = np.convolve(data, k)
    return totalpwr_filter[fs:-fs+1]