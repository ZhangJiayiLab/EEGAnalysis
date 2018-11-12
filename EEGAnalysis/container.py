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


def dircheck(resultdir, expname):
    pass

class CompactDataContainer(object):
    """data container of compact data"""

    def __init__(self, datadir, resultdir, patientname, expname, fs, roi=None):
        warn("use *.edf files for more economy storage", category=DeprecationWarning)

        # meta
        self.name = expname
        self.resultdir = os.path.join(resultdir, patientname)
        self.datadir = os.path.join(datadir, patientname, "EEG")

        dircheck(self.resultdir, self.name)

        # import data
        data = loadmat(os.path.join(self.datadir, "Compact", self.name+".mat"))
        self.channels = data["channels"]
        self.times = data["times"][0, :]
        self.markers = data["markers"]  #TODO: convert to dict

        # constants
        self.fs = fs
        self.iti = self.markers["grating"][0][0][0, 1] - self.markers["grating"][0][0][0, 0]
        if roi == None:
            self.roi = (-2, self.iti)
        else:
            self.roi = roi

    def group_channel_by_marker(self, chidx, markername):
        gap = int((self.roi[1]-self.roi[0])*self.fs)
        marker = self.markers[markername][0][0][0, :]  #TODO conver to dict
        groupdata = np.zeros((len(marker), int((self.roi[1]-self.roi[0])*self.fs)))
        for idx, each in enumerate(marker):
            groupdata[idx, :] = self.channels[chidx, int(np.floor((each+self.roi[0])*self.fs)):int(np.floor((each+self.roi[0])*self.fs))+gap]

        return groupdata

class SplitDataContainer(object):
    """data container of compact data"""
    def __init__(self, sgchdir, chidx, fs, marker_bias=None, markername="grating", _import_date="all", _roi_head=-2):
        self.sgchdir = sgchdir
        self.chidx = chidx
        self.chdir = os.path.join(sgchdir, "ch%03d"%chidx)
        self.fs = fs

        if marker_bias == None:
            self.marker_bias = None
            self.get_marker_bias = lambda expanme: 0
        else:
            self.marker_bias = pd.read_csv(os.path.join(sgchdir, marker_bias))
            self.get_marker_bias = lambda expname: self.marker_bias[self.marker_bias.exp==expname].bias.values[0]

        namepattern = r"(\d{6})-(\d)-(\d{1,2})_ch(\d{3}).mat"

        files = []
        for item in os.listdir(self.chdir):
            if re.match(namepattern, item):
                files.append(item)

        ch_erp = {
            "5": np.zeros((5-_roi_head)*fs),
            "5-1": np.zeros((5-_roi_head)*fs),
            "5-2": np.zeros((5-_roi_head)*fs),
            "5-3": np.zeros((5-_roi_head)*fs),
            "10": np.zeros((10-_roi_head)*fs),
            "10-1": np.zeros((10-_roi_head)*fs),
            "10-2": np.zeros((10-_roi_head)*fs),
            "10-3": np.zeros((10-_roi_head)*fs)
        }

        for eachfile in files:
            try:
                rawdata = rawdata = loadmat(os.path.join(self.chdir, eachfile))
            except Exception:
                warn("empty mat data?!?! %s"%eachfile)
                continue
            
            date, mode, iti, chname = re.findall(namepattern, eachfile)[0]
            expname="%s-%s-%s"%(date, mode, iti)
            
            if _import_date is not "all" and date not in _import_date:
                continue  # skip undesired date

            channel = rawdata["values"][0,:]
            try:
                marker = rawdata["markers"][markername][0][0][0,:]
            except IndexError:
                warn("bad length: %s, file passed"%eachfile, category=RuntimeWarning)
                continue  #give up this file
                
            try:
                marker += self.get_marker_bias(expname)
            except IndexError:
                warn("no marker bias: %s, use 0.0 as default"%eachfile, category=ImportWarning)
                pass

            try:
                epoch = self.group_channel_by_marker(channel, marker, (_roi_head, int(iti)), fs)
            except ValueError:
                warn("bad chunking: %s, file passed"%eachfile, category=RuntimeWarning)
                continue # give up this file

            ch_erp["%s-%s"%(iti, mode)] = np.vstack((ch_erp["%s-%s"%(iti, mode)], epoch))
            ch_erp[iti] = np.vstack((ch_erp[iti], epoch))

        self.ch_erp = {
            "5" : ch_erp["5"][1:, :],
            "5-1" : ch_erp["5-1"][1:, :],
            "5-2" : ch_erp["5-2"][1:, :],
            "5-3" : ch_erp["5-3"][1:, :],
            "10" : ch_erp["10"][1:, :],
            "10-1": ch_erp["10-1"][1:, :],
            "10-2": ch_erp["10-2"][1:, :],
            "10-3": ch_erp["10-3"][1:, :]
        }

    def group_channel_by_marker(self,channel, marker, roi, fs):
        gap = int((roi[1]-roi[0])*fs)
        groupdata = np.zeros((len(marker), gap))
        for idx, each in enumerate(marker):
            starter = int(np.floor((each+roi[0])*fs))
            groupdata[idx, :] = channel[starter:starter+gap]

        return groupdata
