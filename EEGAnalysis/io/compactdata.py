"""
Compact Data manipulation

author: Yizhan Miao
email: yzmiao@protonmail.com
last update: Oct 1 2018
"""

import os, re
from scipy.io import loadmat, savemat
import h5py
import numpy as np
from tqdm import tqdm


def checkcompact(datadir, filename):
    """Check if compact data file exists, return bool."""
    return os.path.isfile(os.path.join(datadir, filename))

def createcompact(datadir, patientname, fs, overwrite=False,
                  match_pattern=r"\d{6}-.*?\.mat"):
    """Create compact data from Spike2 exported mat file.

    Syntax: createcompcat(datadir, patientname, fs, overwrite)

    Keyword arguments:
    datadir       -- (str) data directory path
    patientname   -- (str) subject name
    fs            -- (int) sampling rate
    overwrite     -- (bool) overwrite flag
    match_pattern -- (regx_str)

    Example:
    >>> createcompact("../../Data", "subject1", 2000)

    >>> createcompact("../../Data", "subject2", 2000, overwrite=True)
    """

    _spike2_dir = os.path.join(datadir, patientname, "EEG", "Spike2")
    _compact_dir = os.path.join(datadir, patientname, "EEG", "Compact")

    rawfiles = []
    for item in os.listdir(_spike2_dir):
        if re.match(match_pattern, item) and \
            (not checkcompact(_compact_dir, item) or overwrite):
            rawfiles.append(item)

    print("create compact mat from:\n", rawfiles)
    input("press enter to start ...")
    for idx, item in tqdm(enumerate(rawfiles)):
        with h5py.File(os.path.join(_spike2_dir, item), "r") as f:
            times = list(f["Chan__1"]["times"])[0]
            cueonset = list(f["Memory"]["times"])[0]
            channels = np.zeros((len(f)-2, len(times)))

            for i in f.keys():
                if i == "Memory" or i == "file":
                    continue

                idx = int(np.array(f[i]["title"], dtype="uint8")[4:].tostring())
                channels[idx-1, :] = np.array(f[i]["values"])[0,:]

        length = np.size(channels, 1) / fs
        residual = length - cueonset[-1]
        iti = cueonset[-1] - cueonset[-2]
        entrain_cue = [cueonset[-1] + i*iti+iti for i in range(min(2, int(residual//iti-1)))]
        savemat(os.path.join(_compact_dir, item), {
            "times": times,
            "channels": channels,
            "markers": {
                "grating": cueonset,
                "entrain": entrain_cue}
            })

        if len(entrain_cue) == 0:
            print(item, ": has no valid entrain window!")
