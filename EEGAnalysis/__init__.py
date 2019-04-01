"""
EEGAnalysis module

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 15 2018
"""

from .container import create_epoch_bymarker, create_1d_epoch_bymarker

from .decomposition import *

from .io import *

from .datamanager import DataManager

from .electrodes import Electrodes

from .behavior import get_relative_behavior_time

##### temp #####
import numpy as np
import warnings

def group_consecutive(a, gap=1, killhead=True):
    ''' group consecutive numbers in an array
        modified from https://zhuanlan.zhihu.com/p/29558169'''
    warnings.warn("group_consecutive is no longer maintained.", DeprecationWarning)
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
    warnings.warn("detect_thresh is no longer support, use EEGAnalysis.decomposition.detect_cross_pnt.", DeprecationWarning) 
    if not datadown:  # i.e. data rise
        spike = group_consecutive(np.where(data > thresh)[0], gap=gap)
    else:  # i.e. data down
        spike = group_consecutive(np.where(data < thresh)[0], gap=gap)
    return np.array([item[0] for item in spike])