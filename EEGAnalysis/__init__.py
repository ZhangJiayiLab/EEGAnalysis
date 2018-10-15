__all__ = ["container"]

from .container import CompactDataContainer, SplitDataContainer

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