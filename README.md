# EEG Analysis

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

should at least have 3 vairables:
- `values`: n length 1D vector, as raw data points.
- `times`: n length 1D vector
- `markers`: struct(markername: markervalue)


## index
- [x] EEGAnalysis.CompactDataContainer
- [x] EEGAnalysis.SplitDataContainer
- [x] EEGAnalysis.dwt
- [x] EEGAnalysis.stfft
- [x] EEGAnalysis.power.dwt_power
- [x] EEGAnalysis.phase.dwt_itpc
- [x] EEGAnalysis.io.createcompact

## TODO
- [ ] ERP validation
