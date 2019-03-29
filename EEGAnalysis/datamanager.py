import numpy as np
import pandas as pd
import h5py
from pprint import pprint
import os, re, json, shutil, random
from hashlib import sha256
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython import display

from .io import loadedf
from .container import create_1d_epoch_bymarker
from .decomposition import detect_cross_pnt
from .decomposition.dwt import dwt
from .decomposition.power import dwt_power


def _load_json(filename):
    with open(filename, 'r') as _f:
        _result = json.loads(_f.read())
    return _result


def _save_json(filename, var):
    with open(filename, 'w') as _f:
        _f.write(json.dumps(var))
    return True


class Patient(object):
    '''
    data of single patient and all kinds of manipulations on patient data.
    '''

    def __init__(self, data_dir, patient_id):
        ''' initialization

        arguments:
        - data_dir: root path of the data directory
        - patient_id: patient name or id

        return:
        - `Patient` instance
        '''

        self.id = patient_id
        self._data_dir = data_dir
        self._patient_dir = os.path.join(self._data_dir, patient_id)

        self._raw_dir = os.path.join(self._patient_dir, 'EEG', 'Raw')
        self._raw_config = _load_json(os.path.join(self._raw_dir, 'rawdata.json'))

        self._sgch_dir = os.path.join(self._patient_dir, 'EEG', 'iSplit')
        self._sgch_config = _load_json(os.path.join(self._sgch_dir, 'isplit.json'))

        self._marker_dir = os.path.join(self._patient_dir, 'EEG', 'Marker')
        self._marker_path = os.path.join(self._marker_dir, 'marker.csv')
        if os.path.isfile(self._marker_path):
            _marker = pd.read_csv(self._marker_path)
        else:
            _marker = pd.DataFrame(columns=('file', 'paradigm', 'marker', 'mbias', 'note'))
            _marker.to_csv(self._marker_path, index=False)
        self._marker = pd.read_csv(self._marker_path)

        self._behavior_path = os.path.join(self._marker_dir, 'behavior.csv')
        if os.path.isfile(self._behavior_path):
            _marker = pd.read_csv(self._behavior_path)
        else:
            _marker = pd.DataFrame(columns=('file', 'paradigm', 'marker', 'mbias', 'note'))
            _marker.to_csv(self._behavior_path, index=False)
        self._behavior = pd.read_csv(self._behavior_path)


    def load_raw(self, name=""):
        '''
        load edf format file, with the `name` specified.

        arguments:
        - name: the name of the edf file

        return:
        - `edf` instance
        '''

        _pool = [item['file'] for item in self._raw_config.values() if item['name'] == name]
        if len(_pool) > 1:
            raise ValueError("Duplicated name: \"%s\""%name)
        elif len(_pool) == 0:
            raise ValueError("name not found: \"%s\""%name)
        else:
            return loadedf(_pool[0], 'load_raw')


    def load_isplit(self, chidx, name=None):
        '''
        load isplit format data, with channel index specified.

        arugments:
        - chidx: channel index

        keyword arguments:
        - name: either be string or list of strings, i.e. the names of the target edf files.
                default as None, i.e. import all edf files.

        returns:
        - result :: dict{name: dict{unit: ndarray, value: ndarray, freq: ndarray}}
        '''

        _channel_name = "Channel%03d"%(chidx + 1)
        _hdf5_file = h5py.File(os.path.join(self._sgch_dir, '%s.h5'%_channel_name), 'r')

        result = {}
        if name == None:
            for item in _hdf5_file.values():
                result[item.name[1:]] = {
                    'unit': np.array(item['unit']),
                    'value': np.array(item['value']),
                    'freq': np.array(item['freq']),
                }

        else:
            if isinstance(name, str):
                _name = [name]
            else:
                _name = name

            for item in _name:
                if item in _hdf5_file:
                    result[item] = {
                        'unit': np.array(_hdf5_file[name]['unit']),
                        'value': np.array(_hdf5_file[name]['value']),
                        'freq': np.array(_hdf5_file[name]['freq']),
                    }
                else:
                    raise ValueError("name not found: \"%s\""%name)

        return result

    def check_marker(self):
        '''
        check if the specific name of the edf file has a list of markers.

        return: bool
        '''

        _missed = [item['name'] for item in self._raw_config.values() if item['name'] not in list(self._marker.file)]

        if len(_missed) == 0:
            print('all set.')
            return True
        else:
            print("please use `DataManager.update_marker(patient_id, name, marker_array)` to update markers of the following files:.")
            pprint(_missed)
            return False


    def update_marker(self, name, marker_array, paradigm="", note='', overwrite=False):
        '''
        update the marker list manually

        arguments:
        - name: the name of the edf file
        - marker_array: array of the marker timestamps

        keyword arguments:
        - paradigm: the paradigm tag string
        - note: some note string
        - overwrite: overwrite flag

        return void
        '''

        if name in list(self._marker.file) and not overwrite:
            print("marker alread in record: %s; use `overwrite` flag to overwrite."%name)
            return
        elif name in list(self._marker.file) and overwrite:
            print("overwrite record: %s."%name)
            self._marker = self._marker[self._marker.file != name]
            self._marker = self._marker.append([{'file':name, 'paradigm':paradigm, 'marker':item, 'mbias':'', 'note':note} for item in marker_array])
        else:
            self._marker = self._marker.append([{'file':name, 'paradigm':paradigm, 'marker':item, 'mbias':'', 'note':note} for item in marker_array])

        self._update_marker()
        return

    def update_marker_specification(self, source_dir, copy=True, overwrite=False):
        '''
        copy the marker config file generated by the psychopy programme

        arguments:
        - source_dir: path of the source directory

        keyword arguments:
        - copy: true copy flag
        - overwrite: overwrite flag

        return void
        '''

        _name_pattern = r'\d{6,8}[-_]%s'%self.id
        target_dir = self._marker_dir
        for item in [item for item in os.listdir(source_dir) if re.match(_name_pattern, item)]:
            if os.path.isfile(os.path.join(target_dir, item)) and not overwrite:
                print('file %s is already imported, skip.'%item)
                continue
            elif os.path.isfile(os.path.join(target_dir, item)) and overwrite:
                print('overwriting file %s.'%item)

            shutil.copy(os.path.join(source_dir, item), os.path.join(target_dir, item))

    def _mbias_preview(self, chidx, name, paradigm):
        _marker = self._marker.marker[(self._marker.file == name)&(self._marker.paradigm == paradigm)].values
        _entry = self.load_isplit(chidx, name)
        _freq = int(_entry[name]['freq'])

        _frange = np.logspace(np.log10(1), np.log10(150), 20)
        _tspec = np.linspace(-1, 2, 3 * _freq)

        _chunk = create_1d_epoch_bymarker(_entry[name]['value'], fs=_freq,
                                                 roi=(-1,2), marker=_marker, mbias=0)

        _dwt_result = dwt(data=_chunk, frange=_frange, fs=_freq, reflection=True)
        _pwr = dwt_power(dwtresult=_dwt_result, fs=_freq, zscore=True)

        plt.figure(figsize=(8,6))
        plt.contourf(_tspec, _frange, _pwr, 80, cmap=plt.get_cmap('jet'))
        plt.clim((-5*np.std(_pwr), 5*np.std(_pwr)))
        plt.title('Ch%03d'%(chidx+1))
        plt.tight_layout()
        plt.show()


    def _update_marker(self):
        self._marker.to_csv(self._marker_path, float_format="%.3f", index=False)
        self._marker = pd.read_csv(self._marker_path)

        self._behavior.to_csv(self._behavior_path, float_format="%.3f", index=False)
        self._behavior = pd.read_csv(self._behavior_path)


    def update_mbias(self, name=None, mbias=None, paradigm=None, overwrite=False, n=3):
        '''
        calibrate the time stamp bias manually.
        use this function after you have import the time stamp mannully.
        '''

        if name == None:
            _candidates = np.unique(self._marker.file[self._marker.mbias != None])
        elif isinstance(name, str):
            _candidates = [name]
        elif isinstance(name, list):
            _candidates = name
        else:
            raise ValueError('bad `name`.')

        if paradigm == None:
            _paradigm = np.unique(self._marker.paradigm[self._marker.mbias != None])
        elif isinstance(paradigm, str):
            _paradigm = [paradigm]
        elif isinstance(paradigm, list):
            _paradigm = paradigm
        else:
            raise ValueError('bad `paradigm`.')

        for each in _candidates:
            for _each_paradigm in _paradigm:
                if each in list(self._marker.file[~np.isnan(self._marker.mbias)&(self._marker.paradigm == _each_paradigm)]) and not overwrite:
                    print("mbias alread in record: %s; use `overwrite` flag to overwrite."%each)
                    continue
                elif each in list(self._marker.file[~np.isnan(self._marker.mbias)&(self._marker.paradigm == _each_paradigm)]) and overwrite:
                    print("overwriting record: %s;"%each)

                if len(self._marker.marker[(self._marker.paradigm == _each_paradigm)&(self._marker.file == each)]) == 0:
                    continue
                print(each+'-'+_each_paradigm)
                _count = 0
                _mbias = []
                while _count < n:
                    _idx = random.randrange(len(self._sgch_config.keys()))
                    self._mbias_preview(_idx, each, _each_paradigm)
                    _input = input('type mbias, or \'.\' to skip this channel:')
                    if _input == '.':
                        pass
                    else:
                        _mbias.append(float(_input))
                        _count = _count + 1
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

                self._marker.mbias[self._marker.file == each] = np.mean(_mbias)
                plt.close()
                self._update_marker()


    def _get_chidx(self, channel_label):
        '''
        convert the channel label in the edf file to the channel index.
        '''

        if 'chidx' not in self._sgch_config.keys():
            self._sgch_config['chidx'] = {}

        if re.match(r'(POL|EDF) ((DC|BP|EKG|EMGL|EMGR)\d{1,2}|Annotations)', channel_label):
            return -1

        if channel_label not in self._sgch_config['chidx'].keys():
            self._sgch_config['chidx'][channel_label] = len(self._sgch_config['chidx'])
            self._update_config()

        return self._sgch_config['chidx'][channel_label]

    def _update_config(self):
        '''
        update and overwrite the config file of isplit and rawdata directories.
        '''

        _save_json(os.path.join(self._sgch_dir, 'isplit.json'), self._sgch_config)
        _save_json(os.path.join(self._raw_dir, 'rawdata.json'), self._raw_config)
        return


    def create_isplit(self, compression_level=4, overwrite=False):
        '''
        create and update isplit files from edf raw data

        keyword arguments:
        - compression_level: the level of compression, default as 4.
            0 as no compression and 10 as the highest compression level.
        - overwrite: the overwrite flag

        return void
        '''

        pbar = tqdm(total=len(self._raw_config.keys()))

        for raw_file, raw_item in self._raw_config.items():
            _edf_data = loadedf(raw_item['file'], 'create_isplit')

            for _idx in range(_edf_data.nchannel):

                chidx = self._get_chidx(_edf_data.channelLabels[_idx])
                if chidx == -1:
                    continue

                _channel_name = 'Channel%03d'%(chidx+1)

                if not _channel_name in self._sgch_config.keys():
                    self._sgch_config[_channel_name] = []

                _sha = sha256(_edf_data.data[chidx]).hexdigest()
                if _sha in self._sgch_config[_channel_name] and not overwrite:
                    continue

                _hdf5_file = h5py.File(os.path.join(self._sgch_dir, '%s.h5'%_channel_name), 'a')
                if raw_item['name'] not in _hdf5_file:
                    _hdf5_file.create_group(raw_item['name'])

                _hdf5_file.create_dataset(name='%s/unit'%raw_item['name'], data=_edf_data.physical_unit[chidx])
                _hdf5_file.create_dataset(name='%s/value'%raw_item['name'], data=_edf_data.data[chidx], compression="gzip", compression_opts=compression_level)
                _hdf5_file.create_dataset(name="%s/freq"%raw_item['name'], data=_edf_data.fs)

                _hdf5_file.close()
                self._sgch_config[_channel_name].append(_sha)
            pbar.update(1)

        _save_json(os.path.join(self._sgch_dir, 'isplit.json'), self._sgch_config)
        pbar.close()
        return


    def update_DC_marker(self, overwrite=False, mapping={'POL DC10': 'marker'}, thresh=3):
        '''
        automatic updating marker list.

        keyword arguments:
        - overwrite: overwrite flag
        - mapping: mapping of the marker channels [future]

        return void
        '''

        for _target_ch, _marker_name in mapping.items():
            _marker_path = os.path.join(self._marker_dir, '%s.csv'%_marker_name)
            try:
                _marker_file = pd.read_csv(_marker_path)
            except FileNotFoundError:
                _marker_file = pd.DataFrame(columns=['file','paradigm','marker','mbias','note'])

            for item in self._raw_config.values():
                if item['name'] in list(_marker_file.file) and not overwrite:
                    print('alreday exist the markers of %s, skip.'%(item['name']))
                    continue
                elif item['name'] in list(_marker_file.file) and overwrite:
                    print('overwrite the markers of %s'%(item['name']))
                else:
                    pass

                _edf = loadedf(item['file'], 'parse marker')

                try:
                    _marker_ch = np.where([True if _item == _target_ch else False for _item in _edf.channelLabels])[0][0]
                except IndexError:
                    print('file %s has no target DC channels: %s'%(item['name'], _target_ch))
                    continue

                try:
                    _marker_trace = _edf.data[_marker_ch] * _edf.physical_unit[_marker_ch] / 1e6  # unit as Volt
                    _marker_time = np.array(detect_cross_pnt(_marker_trace, thresh, gap=_edf.fs)) / _edf.fs
                except IndexError:
                    print('%s marker of file %s not detected!'%(_marker_name, item['name']))
                    continue


                _marker_data = [{'file':item['name'], 'paradigm':'', 'marker':_item, 'mbias':'0','note':''}
                               for _item in _marker_time]
                _marker_file = _marker_file.append(_marker_data)

                _marker_file.to_csv(_marker_path, float_format="%.3f", index=False)
                print("%s for %s marker of %s: %d"%(_target_ch, _marker_name, item['name'], _marker_ch))

        print('please reload Patient class to use updated marker.')


    def get_marker(self, marker='marker', dtype=None, **filt_param):
        '''
        get the marker array for specific name and paradigm.
        
        default headers include "file", "paradigm", "marker", 
        "mbias", and "note"

        argument:
        - marker: the name of the marker file [default: marker]
        - dtype: specify the pandas data types when read in the marker file [default: None]
        - **filt_param: querying items according to your input parameters [default: None]

        return:
        - marker_arr: ndarray of marker timestamps.
        - pandas.DataFrame: only when `filt_param` has zero length.
        '''
        
        _marker_path = os.path.join(self._marker_dir, marker + '.csv')
        _marker_sheet = pd.read_csv(_marker_path, dtype, engine='python')
        
        _marker_filter = np.ones(len(_marker_sheet), dtype='bool')
        
        if len(filt_param) == 0:
            return _marker_sheet
        else:
            for filtername, filtervalue in filt_param.items():
                _marker_filter = _marker_filter & (_marker_sheet[filtername] == filtervalue)

            return _marker_sheet.marker[_marker_filter].values

class DataManager(object):
    '''
    data manager for all kinds of data formats.
    '''

    def __init__(self, data_dir):
        super().__init__()
        self._data_dir = data_dir


    def get_patient(self, patient_id):
        self.current_patient = Patient(self._data_dir, patient_id)
        return self.current_patient


    def create_patient(self, patient_id):
        '''
        create all the subfolders and config files for a new patient.
        if the patient exists, create all the file that were missing.
        '''

        _patient_dir = os.path.join(self._data_dir, patient_id)
        _new_dirs = [
            self._data_dir,
            _patient_dir,
            os.path.join(_patient_dir, 'EEG'),
            os.path.join(_patient_dir, 'EEG', 'Raw'),
            os.path.join(_patient_dir, 'EEG', 'iSplit'),
            os.path.join(_patient_dir, 'EEG', 'Marker'),
            os.path.join(_patient_dir, 'Image'),
        ]

        _new_configs = [
            os.path.join(_patient_dir, 'EEG', 'Raw', 'rawdata.json'),
            os.path.join(_patient_dir, 'EEG', 'iSplit', 'isplit.json'),
        ]

        [os.mkdir(item) for item in _new_dirs if not os.path.isdir(item)]
        [os.chmod(item, 0o775) for item in _new_dirs if not os.path.isdir(item)]
        for item in _new_configs:
            if not os.path.isfile(item):
                with open(item, 'w') as _f:
                    _f.write("{}")

        self.current_patient = Patient(self._data_dir, patient_id)
        return self.current_patient


    def has_patient(self, patient_id):
        _patient_dir = os.path.join(self._data_dir, patient_id)
        return os.path.isdir(_patient_dir)


    def update_raw_to_patient(self, patient_id, raw_dir, copy=True, ext='.edf', overwrite=False):
        if not self.has_patient(patient_id):
            self.create_patient(patient_id)

        _patient_dir = os.path.join(self._data_dir, patient_id)
        _raw_dir = os.path.join(_patient_dir, 'EEG', 'Raw')
        # _sgch_dir = os.path.join(_patient_dir, 'EEG', 'iSplit')

        _raw_config = _load_json(os.path.join(_raw_dir, 'rawdata.json'))

        for item in tqdm(os.listdir(raw_dir)):
            if (not re.match(r'.*?'+ext, item)) or (item in _raw_config.keys() and not overwrite):
                continue

            if copy:
                shutil.copy(os.path.join(raw_dir, item), os.path.join(_raw_dir, item))
                _store_dir = _raw_dir
            else:
                _store_dir = raw_dir

            # _temp = open(os.path.abspath(os.path.join(_store_dir, item)), 'r')
            _temp = loadedf(os.path.abspath(os.path.join(_store_dir, item)), 'test')
            _raw_config[item] = {'file':os.path.abspath(os.path.join(_store_dir, item)),
                                 'name':os.path.splitext(item)[0],
                                 'ext': ext,
                                 'sha256': sha256(_temp.data).hexdigest()
                                }
            # _temp.close()

        _save_json(os.path.join(_raw_dir, 'rawdata.json'), _raw_config)
        self.current_patient = Patient(self._data_dir, patient_id)
        return self.current_patient


    def create_isplit(self, patient_id, compression_level=0):
        '''
        deprecated.
        '''

        _patient_dir = os.path.join(self._data_dir, patient_id)
        _raw_dir = os.path.join(_patient_dir, 'EEG', 'Raw')
        _sgch_dir = os.path.join(_patient_dir, 'EEG', 'iSplit')

        _raw_config = _load_json(os.path.join(_raw_dir, 'rawdata.json'))
        _sgch_config = _load_json(os.path.join(_sgch_dir, 'isplit.json'))

        _temp = loadedf(list(_raw_config.values())[0]['file'], 'check_values')
        pbar = tqdm(total=len(_raw_config.keys()) * _temp.nchannel)
        _temp = None


        for raw_file, raw_item in _raw_config.items():
            _edf_data = loadedf(raw_item['file'], 'create_isplit')

            for chidx in range(_edf_data.nchannel):
                pbar.update(1)
                _channel_name = 'Channel%03d'%(chidx+1)

                if not _channel_name in _sgch_config.keys():
                    _sgch_config[_channel_name] = []

                _sha = sha256(_edf_data.data[chidx]).hexdigest()
                if _sha in _sgch_config[_channel_name]:
                    continue

                _hdf5_file = h5py.File(os.path.join(_sgch_dir, '%s.h5'%_channel_name), 'a')
                if raw_item['name'] not in _hdf5_file:
                    _hdf5_file.create_group(raw_item['name'])

                _hdf5_file.create_dataset(name='%s/unit'%raw_item['name'], data=_edf_data.physical_unit[chidx])
                _hdf5_file.create_dataset(name='%s/value'%raw_item['name'], data=_edf_data.data[chidx], compression="gzip", compression_opts=compression_level)
                _hdf5_file.create_dataset(name="%s/freq"%raw_item['name'], data=_edf_data.fs)

                _hdf5_file.close()
                _sgch_config[_channel_name].append(_sha)

        _save_json(os.path.join(_sgch_dir, 'isplit.json'), _sgch_config)
        pbar.close()

        self.current_patient = Patient(self._data_dir, patient_id)
        return self.current_patient



