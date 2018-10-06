"""
Split Data manipulation

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 4 2018
"""

import os, re
from scipy.io import loadmat, savemat
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
# import h5py

def _stepCreateSplitData(_input):
    exportmat, sourcemat, chidx= _input

    rawmat = loadmat(sourcemat)
    savemat(exportmat, {
            "values": rawmat['channels'][chidx, :],
            "times":  (rawmat["times"][0,0], rawmat["times"][0,-1]),
            "markers": rawmat["markers"],
            "fs": 1 / (rawmat["times"][0,1] - rawmat["times"][0,0])
        })

def createSplitData(datadir, patientname, chrange=range(123), pn=3, overwrite=False):
    """Split compact data, with multiprocessing module."""
    _datadir = os.path.join(datadir, patientname, "EEG", "Compact")
    _exportdir = os.path.join(datadir, patientname, "EEG", "SgCh")

    files = []
    matching_pattern = r"\d{6}-.*?\.mat"
    for item in os.listdir(_datadir):
        if re.match(matching_pattern, item):
            files.append(os.path.splitext(item)[0])

    map_args = []
    for chidx in chrange:
        exportfiledir = os.path.join(_exportdir, "ch%03d"%(chidx))
        if not os.path.isdir(exportfiledir):
            os.mkdir(exportfiledir)
        for each_source in files:
            exportmat = os.path.join(exportfiledir, each_source+"_ch%03d.mat"%chidx)
            sourcemat = os.path.join(_datadir, each_source+'.mat')
            if os.path.isfile(exportmat) and not overwrite:
                continue
            else:
                map_args.append((exportmat, sourcemat, chidx))

    with Pool(processes=pn) as p:
        with tqdm(total=len(map_args)) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(_stepCreateSplitData, map_args))):
                pbar.update()
