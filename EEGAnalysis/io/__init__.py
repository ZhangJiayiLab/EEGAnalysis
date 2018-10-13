__all__ = [
    "compactdata", "splitdata"
]

from .edfdata import EDFData

def loadedf(filename, expname):
    return EDFData(filename, expname)