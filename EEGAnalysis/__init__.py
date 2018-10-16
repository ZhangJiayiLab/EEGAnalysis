"""
EEGAnalysis module

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 15 2018
"""

from .container import CompactDataContainer, SplitDataContainer, iSplitContainer, create_epoch_bymarker

from .decomposition import *

from .io import *

def loadsplitdata(sgchdir, chidx, fs, markername="grating", marker_bias=None, _import_date="all", _roi_head=-2):
    """wrapper for loading split data
    
    Syntax: SplitDataContainer = loadsplitdata(sgchdir, chidx, fs, markername, marker_bias, _import_date, _roi_head)
    
    Key Arguments:
    sgchdir       -- (str) path of SgCh folder
    chidx         -- (int) target channel index (start from 0)
    fs            -- (float) sampling frequency
    markername    -- (str) marker name [default: "grating"]
    marker_bias   -- (str) file name of marker_bias.csv [default: None]
    _import_date  -- (list) explicitly import data from certain date [default: "all"]
    _roi_head     -- (float) the roi range would be (_roi_head, ITI);
                     specially, for "grating", _roi_head should be larger than -3 [default: -2]
    
    Notes:
    # check the ch_erp shape to validate the result.
    # for each date, each mode, there would be 20 grating trials, 
    # and less than 2 entrain trials (few may only have
    # one or none entrain trials).

    # ch_erp now has concatenated tials of each different paradigms
    # 5: for all three paradigms
    # 5-1, 5-2, 5-3 for each paradigm respectively
    # the same for 10, 10-1, 10-2, 10-3
    """
    return SplitDataContainer(sgchdir, chidx, fs, marker_bias, markername, _import_date, _roi_head)


##### temp #####
import numpy as np

def group_consecutive(a, gap=1, killhead=True):
    ''' group consecutive numbers in an array
        modified from https://zhuanlan.zhihu.com/p/29558169'''
    if len(a) == 0:
        return []
    if killhead and a[0] == 0:
        try:
            skip = np.where(np.diff(a) > gap)[0][0]
            return np.split(a, np.where(np.diff(a)[skip:] > gap)[0] + skip + 1)[1:]
        except IndexError:
            return []
    else:
        return np.split(a, np.where(np.diff(a) > gap)[0] + 1)
    
def detect_thresh(data, thresh, gap=1, datadown=False):
    """detect datarise or datadown index by threshold
    
    Syntax: Idx = detect_thresh(data, thresh, gap, datarise)
    
    Key Arguments:
    data     -- (np.array) 1D data array
    thresh   -- (number) threshold for data
    gap      -- (int) consecutive gap [default: 1]
    datadown -- (bool) check for datadown (True) or datarise (False)
                [default: False]
    
    Return:
    Idx      -- (np.array) indices of the thresh points
    
    """
    if not datadown:  # i.e. data rise
        spike = group_consecutive(np.where(data > thresh)[0], gap=gap)
    else:  # i.e. data down
        spike = group_consecutive(np.where(data < thresh)[0], gap=gap)
    return np.array([item[0] for item in spike])