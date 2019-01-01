"""
EEG analysis data container

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 15 2018
"""

import os, re
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
from warnings import warn

def create_epoch_bymarker(data, marker, roi, fs, mbias=0):
    gap = int(np.ceil((roi[1] - roi[0]) * fs))
    result = np.zeros((np.size(data, 0), gap, len(marker)), dtype=data.dtype)
    for midx, eachm in enumerate(marker):
        start = int(np.floor((eachm + roi[0] + mbias) * fs))
        result[:, :, midx] = data[:, start:start+gap]
    return result

def create_1d_epoch_bymarker(data, marker, roi, fs, mbias=0):
    gap = int(np.ceil((roi[1] - roi[0]) * fs))
    result = np.zeros((len(marker), gap), dtype=data.dtype)
    for midx, eachm in enumerate(marker):
        start = int(np.floor((eachm + roi[0] + mbias) * fs))
        result[midx, :] = data[start:start+gap]
    return result


class iSplitContainer(object):
    def __init__(self, datadir, chidx):
        warn('.mat backend isplit will not be supperted in the future, please use `data manager` to create and load isplit data.(hdf5 backend)', DeprecationWarning)
        self.chfilename = os.path.join(datadir, "sgch_channel_%03d.mat"%chidx)
        
        rawmat = loadmat(self.chfilename)
        self.edfnames = rawmat["edfnames"]
        self.values = dict([(self.edfnames[idx], item[0]) for idx, item in enumerate(rawmat["values"][0])])
        self.physicalunit = dict([(self.edfnames[idx], item) for idx, item in enumerate(rawmat["physicalunit"][0])])
        self.samplingfrequency = dict([(self.edfnames[idx], item) for idx, item in enumerate(rawmat["samplingfrequency"][0])])
        
    def _chunk_bymarkername(self, marker, markername, roi, mbias):
        chunk_names = []
        
        result = []
        for eachname in self.edfnames:
            markerlist = marker[(marker.id==eachname) & (marker.mname==markername)].marker.values
            _temp = create_1d_epoch_bymarker(self.values[eachname], markerlist, roi, 
                                  self.samplingfrequency[eachname], mbias=mbias)
            result.append(_temp)
            chunk_names.append(eachname)
        
        return chunk_names, result
    
    def chunk_bymarkername(self, marker, markername, roi, mbias=0):
        _, r = self._chunk_bymarkername(marker, markername, roi, mbias=mbias)
        rr = r[1]
        for item in r[2:]:
            rr = np.vstack((rr, item))
        return rr


