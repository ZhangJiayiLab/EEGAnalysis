import numpy as np
from .io import loadedf

import os, re, json, shutil
from hashlib import sha256
import h5py
from tqdm import tqdm

def _load_json(filename):
    with open(filename, 'r') as _f:
        _result = json.loads(_f.read())
    return _result

def _save_json(filename, var):
    with open(filename, 'w') as _f:
        _f.write(json.dumps(var))
    return True

class DataManager(object):
    def __init__(self, data_dir):
        super().__init__()
        self._data_dir = data_dir
    
    def create_patient(self, patient_id):
        _patient_dir = os.path.join(self._data_dir, patient_id)
        _new_dirs = [
            self._data_dir,
            _patient_dir,
            os.path.join(_patient_dir, 'EEG'),
            os.path.join(_patient_dir, 'EEG', 'Raw'),
            os.path.join(_patient_dir, 'EEG', 'iSplit'),
            os.path.join(_patient_dir, 'Image'),
        ]
        
        _new_configs = [
            os.path.join(_patient_dir, 'EEG', 'Raw', 'rawdata.json'),
            os.path.join(_patient_dir, 'EEG', 'iSplit', 'isplit.json'),
        ]
        
        [os.mkdir(item) for item in _new_dirs if not os.path.isdir(item)]
        for item in _new_configs:
            if not os.path.isfile(item):
                with open(item, 'w') as _f:
                    _f.write("{}")
        
    
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
        return _raw_config
    
    def create_isplit(self, patient_id, compression_level=0):
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
            
                _hdf5_file.close()
                _sgch_config[_channel_name].append(_sha)
        
        _save_json(os.path.join(_sgch_dir, 'isplit.json'), _sgch_config)
        pbar.close()
        
        return _sgch_config
    
    def load_rawconfig(self, patient_id):
        _patient_dir = os.path.join(self._data_dir, patient_id)
        _raw_dir = os.path.join(_patient_dir, 'EEG', 'Raw')
        _raw_config = _load_json(os.path.join(_raw_dir, 'rawdata.json'))
        return _raw_config
    
    def load_isplitconfig(self, patient_id):
        _patient_dir = os.path.join(self._data_dir, patient_id)
        _sgch_dir = os.path.join(_patient_dir, 'EEG', 'iSplit')
        _sgch_config = _load_json(os.path.join(_sgch_dir, 'isplit.json'))
        return _sgch_config
    
    def load_raw(self, patient_id, name=""):
        _patient_dir = os.path.join(self._data_dir, patient_id)
        _raw_dir = os.path.join(_patient_dir, 'EEG', 'Raw')
        _raw_config = _load_json(os.path.join(_raw_dir, 'rawdata.json'))
        
        _pool = [item['file'] for item in _raw_config.values() if item['name'] == name]
        if len(_pool) > 1:
            raise ValueError("Duplicated name: \"%s\""%name)
        elif len(_pool) == 0:
            raise ValueError("name not found: \"%s\""%name)
        else:
            return loadedf(_pool[0], 'load_raw')
    
    def load_isplit(self, patient_id, chidx, name=None):
        _patient_dir = os.path.join(self._data_dir, patient_id)
        _sgch_dir = os.path.join(_patient_dir, 'EEG', 'iSplit')
        _sgch_config = _load_json(os.path.join(_sgch_dir, 'isplit.json'))
        
        _channel_name = "Channel%03d"%(chidx + 1)
        _hdf5_file = h5py.File(os.path.join(_sgch_dir, '%s.h5'%_channel_name), 'r')
        
        result = {}
        if name == None:
            for item in _hdf5_file.values():
                result[item.name[1:]] = {
                    'unit': np.array(item['unit']),
                    'value': np.array(item['value']),
                }
        
        else:
            if name in _hdf5_file:
                result[name] = {
                    'unit': np.array(_hdf5_file[name]['unit']),
                    'value': np.array(_hdf5_file[name]['value']),
                }
            else:
                raise ValueError("name not found: \"%s\""%name)
                
        return result
        
        
    