# EEG Analysis

## Get Started
1. re-organize your data folders into [required structure](#Folder-Structure)
2. install `python3`
3. install required python modules by command `python -m pip install -r requirements.txt`
4. install jupyter notebook by command `python -m pip install jupyter jupyterlab`
5. prepare your data according to the data specification or use
   the EEGAnalysis.io methods to create your data
6. read and learn with the `EA_demo.ipynb`

## Folder Structure
```
+Data
|-+Patient 1
| |-+EEG
| | |-+Compact
| | | |-compact-180829-1-5.mat
| | | |- ...
| | |
| | |-+Layout
| | | |-Patient 1-layout.csv
| | | |- ...
| | |
| | |-+Raw
| | | |-180829-1-5.???
| | | |-...
| | |
| | |-+Spike2
| | | |-180829-1-5.mat
| | | |- ...
| | |
| | |-+SgCh
| | | |-marker_bais.csv
| | | |-+ch000
| | | | |-180829-1-5_ch000.mat
| | | | |-...
| | | |
| | | |-...
| |
| |-+Imaging
| | |-+Pre-op
| | | |-+DICOM
| | |
| | |-+Post-op
| | | |-+DICOM
|
|-+Patient 2

+Result
|-+Patient 1
| |-+tf_domain
| | |-180829-1-5.mat
| | |- ...
| |
| |-+

```

## Compact data specification

create by `EEGAnalysis.io.compactdata.createcompact`

import by `EEGAnalysis.CompactDataContainer`

using `mat` format.
(for scipy.io.loadmat, please use `mat` lower than `v7.3`)

should at least have 3 variables:
- `channels`: m\*n matrix, columns as channels from `ch 1` to `ch m`, rows as raw data points.
- `times`: n length 1D vector
- `markers`: struct(markername: markervalue)

## Split data specification

create by `EEGAnalysis.io.splitdata.createsplit`

import by `EEGAnalysis.SplitDataContainer`

using `mat` format
(for scipy.io.loadmat, please use `mat` lower than `v7.3`)

should at least have 2 vairables:
- `values`: n length 1D vector, as raw data points.
- `markers`: struct(markername: markervalue) (TODO: to deprive)
- `times` (no need to declare): take the first data point as 0.0

## TODO
- [ ] documentation
- [ ] use edf file for patient 2
- [ ] transform to Int16 data sets
- [x] ERP validation
