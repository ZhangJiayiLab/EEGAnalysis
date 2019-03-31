"""
decomposition analysis by short time fast fourier tranform

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 1 2018
"""

import numpy as np
# from numpy import jit  #TODO: numba acceleration

## Taper
def han(timepoints):
    """Han Taper function. i.e. shifted half cosine."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, timepoints))


## stfft
def stfft(data, nwindow, noverlap, fs, taper=han, rho=2):
    """Perform short time fast fourier transformation

    Syntax: (Stfft, Tspec) = stfft(data, nwindow, noverlap, fs, taper, rho)

    Keyword arguments:
    data     -- (numpy.ndarray) 1D or 2D array. for 2D array, columns as
                observations, rows as raw data.
    nwindow  -- (int) fft window size, as number of data points.
    noverlap -- (int) overlap points between two windows
    fs       -- (int) sampling frequency
    taper    -- (function) taper function, take <int> as input. [default: han]
    rho      -- (int) density of fft readout frequency,
                relative to sampling frequency. [default: 2]

    Return:
    Stfft    -- (numpy.ndarray, dtype="complex") stfft result, in complex form.
    Tspec    -- (numpy.ndarray) time point of each stfft time bin.

    """

    step = nwindow - noverlap
    start = nwindow // 2
    nstep = (np.size(data, -1) - nwindow) // step

    Tspec = np.linspace(start, step * nstep, nstep) / fs
    Stfft = np.zeros((np.size(data, 0), rho * 500, nstep))

    taperl = taper(nwindow)
    for idx in range(nstep):
        temp = data[(slice(None), slice(idx*step, idx*step+nwindow))] * taperl
        entry = np.fft.fft(temp, n=rho*fs)
        Stfft[(slice(None), slice(None), idx)] = entry

    return Stfft, Tspec
