"""
total power EEG Analysis

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 1 2018
"""

import numpy as np

def dwt_power(dwtresult, fs,  zscore=True, baseline=None):
    """calcuate the total power from the result of dwt

    Syntax: Pxx = dwt_power(dwtresult, zscore, dbcalibration)

    Keyword arguments:
    dwtresult -- (numpy.ndarray, dtype="complex") the 3D result from dwt function
    fs        -- (int) sampling rate
    zscore    -- (bool) normalize the db result by z-score normalization
                 [default: True]
    baseline  -- (tuple(float, float)) normalize the db result
                 by baseline normalization
                 [default: None]

    Return:
    Pxx       -- (numpy.ndarray) total power
    """

    # generate power and averaged across tirlas (axis 1)
    raw_pxx = np.mean(np.abs(dwtresult) ** 2.0, 1)
    
    if baseline != None:
        starter = int(baseline[0]*fs)
        gap = int((baseline[1] - baseline[0])*fs)
        _base = raw_pxx[:, starter:(starter+gap)]
    else:
        _base = raw_pxx

    if not zscore:
        _baseline = np.mean(_base, 1)
        _baseline = np.reshape(_baseline, (np.size(_baseline, 0), 1))
        Pxx = 10 * np.log10(raw_pxx / _baseline)
    elif zscore:
        mu = np.reshape(np.mean(_base, 1), (np.size(_base, 0), 1))
        sig = np.reshape(np.std(_base, 1), (np.size(_base, 0), 1))
        Pxx = (raw_pxx - mu) / sig
    else:
        Pxx = np.log10(raw_pxx)

    return Pxx
