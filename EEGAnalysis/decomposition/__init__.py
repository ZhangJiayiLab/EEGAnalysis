__all__ = [
        "stfft", "dwt", "phase", "power", "filter", "detect_cross_pnt"
]

from .stfft import stfft
# from .dwt import dwt
from .phase import dwt_itpc
from .power import dwt_power
# import hilbert  #TODO: hilbert transform
from .filter import gaussianwind

import numpy as np

def detect_cross_pnt(arr, thr, way='up', gap=1):
    """
    detect the data rise/down point, returns the index of the 
    point right above the threshold.
    
    arguments:
    - arr: data array (1d)
    - thr: threshold (scale)
    
    key arguments:
    - way: either be "up" or "down", for data rise/ data down respectively.
    - gap: the least points between two valid markers.
    
    returns:
    - _marker_idx: index array (1d)
    """
    
    _idx, = np.where(arr > thr)
    _idx_diff, = np.where(np.diff(_idx) > 1)
    _idx_repo = np.hstack((_idx[0], _idx[_idx_diff], _idx[_idx_diff+1], _idx[-1]))
    
    if way == 'up':
        _check = lambda x, i: x[i-1] < thr < x[i] #< x[i+1]
    elif way =='down':
        _check = lambda x, i: x[i-1] > x[i] > x[i+1]
    else:
        raise ValueError("unknown `way` value.")
    
    _idx_repo.sort()
    _result = []
    _previous = -9999
    for idx in _idx_repo:
        try:
            if _check(arr, idx) and (idx - _previous > gap):
                _result.append(idx)
                _previous = idx
        except IndexError:
            pass
    
    return _result