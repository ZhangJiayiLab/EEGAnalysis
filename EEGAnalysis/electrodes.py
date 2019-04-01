import pandas as pd
import numpy as np
import os
import h5py


def _mni2tal(mni):
    
    rotn = np.array([[1,0,0,0],
                     [0,0.9988,0.05,0],
                     [0,-0.05,0.9988,0],
                     [0,0,0,1]])
    upz = np.array([[0.99,0,0,0],
                    [0,0.97,0,0],
                    [0,0,0.92,0],
                    [0,0,0,1]])
    downz = np.array([[0.99,0,0,0],
                      [0,0.97,0,0],
                      [0,0,0.84,0],
                      [0,0,0,1]])
    
    inpoints = np.hstack((mni, np.ones((np.size(mni,0), 1)))).T
    _post = inpoints[2, :] < 0

    inpoints[:,  _post] = np.matmul(np.matmul(rotn, downz),inpoints[:,  _post])
    inpoints[:, ~_post] = np.matmul(np.matmul(rotn, upz),  inpoints[:, ~_post])
    
    return inpoints[:3, :].T


class Electrodes(object):
    
    def __init__(self, datadir, patient_id, contact_width=2, gap_width=1.5):
        
        self.patient_id = patient_id
        self.coord = {}
        self._shank = {}
        self._image_dir = os.path.join(datadir, patient_id, 'Image')
        self._sheet = None
        self._contact_width = contact_width
        self._gap_width = gap_width
        
        
    def create_layout(self):
        while True:
            _shank = input('new shank name [enter "q" to quit]:')
            if _shank == 'q':
                break
            
            print('enter MNI coord of shank tip')
            _shank_tip_mni_x = float(input('shank tip MNI - X:'))
            _shank_tip_mni_y = float(input('shank tip MNI - Y:'))
            _shank_tip_mni_z = float(input('shank tip MNI - Z:'))
            _shank_tip_mni = np.array([_shank_tip_mni_x, _shank_tip_mni_y, _shank_tip_mni_z]).reshape((1,3))
            
            print('enter MNI coord of shank tail')
            _shank_tail_mni_x = float(input('shank tail MNI - X:'))
            _shank_tail_mni_y = float(input('shank tail MNI - Y:'))
            _shank_tail_mni_z = float(input('shank tail MNI - Z:'))
            _shank_tail_mni = np.array([_shank_tail_mni_x, _shank_tail_mni_y, _shank_tail_mni_z]).reshape((1,3))
            
            print('enter CT coord of shank tip')
            _shank_tip_ct_x = float(input('shank tip CT - X:'))
            _shank_tip_ct_y = float(input('shank tip CT - Y:'))
            _shank_tip_ct_z = float(input('shank tip CT - Z:'))
            _shank_tip_ct = np.array([_shank_tip_ct_x, _shank_tip_ct_y, _shank_tip_ct_z]).reshape((1,3))
            
            print('enter CT coord of shank tail')
            _shank_tail_ct_x = float(input('shank tail CT - X:'))
            _shank_tail_ct_y = float(input('shank tail CT - Y:'))
            _shank_tail_ct_z = float(input('shank tail CT - Z:'))
            _shank_tail_ct = np.array([_shank_tail_ct_x, _shank_tail_ct_y, _shank_tail_ct_z]).reshape((1,3))
            
            _shank_channel_n = int(input('enter total number of channels in shank %s: '%_shank))
            
            _L = np.sqrt(np.sum((_shank_tail_ct - _shank_tip_ct)**2))  # physical length, L
            _rho = np.array([ ((self._contact_width/2) + (self._contact_width+self._gap_width)*item)/_L for item in range(_shank_channel_n) ]).reshape((-1, 1))
            _MNI =  _rho * (_shank_tail_mni - _shank_tip_mni) + _shank_tip_mni
            
            self.coord[_shank] = _MNI
            self._shank[_shank] = {'CT':{'tail':_shank_tail_ct, 'tip':_shank_tip_ct},
                                   'MNI':{'tail':_shank_tail_mni, 'tip':_shank_tip_mni},
                                   'n':_shank_channel_n,
                                   'L':_L,
                                   'rho':_rho}
    
    def export_hdf5(self, to=None, filename=None):
        if to == None:
            to = self._image_dir
        
        if filename == None:
            filename = self.patient_id + '_electrodes.h5'
        
        _hdf_path = os.path.join(to, filename)
        with h5py.File(_hdf_path, 'w') as _f:
            for _shank, _data in self._shank.items():
                _tmp = _f.create_group(_shank)
                _tmp.create_dataset('n', data=_data['n'])
                _tmp.create_dataset('L', data=_data['L'])
                _tmp.create_dataset('rho', data=_data['rho'])

                _tmp_ct = _tmp.create_group('CT')
                _tmp_mni = _tmp.create_group('MNI')

                _tmp_ct.create_dataset('tail', data=_data['CT']['tail'])
                _tmp_ct.create_dataset('tip', data=_data['CT']['tip'])

                _tmp_mni.create_dataset('tail', data=_data['MNI']['tail'])
                _tmp_mni.create_dataset('tip', data=_data['MNI']['tip'])
                
    
    def restore_hdf5(self, to=None, filename=None):
        if to == None:
            to = self._image_dir
        
        if filename == None:
            filename = self.patient_id + '_electrodes.h5'
        
        _hdf_path = os.path.join(to, filename)
        self._shank = {}
        with h5py.File(_hdf_path, 'r') as _f:
            for _shank in _f.keys():
                _tmp = {}
                
                _tmp['n'] = np.array(_f[_shank]['n'])
                _tmp['L'] = np.array(_f[_shank]['L'])
                _tmp['rho'] = np.array(_f[_shank]['rho'])
                
                _tmp['CT'] = {'tip':np.array(_f[_shank]['CT']['tip']), 
                              'tail':np.array(_f[_shank]['CT']['tail'])}
                _tmp['MNI'] = {'tip':np.array(_f[_shank]['MNI']['tip']), 
                               'tail':np.array(_f[_shank]['MNI']['tail'])}
                
                _L = np.sqrt(np.sum((_tmp['CT']['tail'] - _tmp['CT']['tip'])**2))  # physical length, L
                _rho = np.array([ ((self._contact_width/2) + (self._contact_width+self._gap_width)*item)/_L for item in range(_tmp['n']) ]).reshape((-1, 1))
                _MNI =  _tmp['rho'] * (_tmp['MNI']['tail'] - _tmp['MNI']['tip']) + _tmp['MNI']['tip']
                
                self._shank[_shank] = _tmp
                self.coord[_shank] = _MNI
    
    def append_new_shank(self):
        pass
        
    def export_csv(self, to=None, filename=None):
        if filename == None:
            filename = self.patient_id + '.csv'
        
        if to == None:
            to = self._image_dir
        
        _data = []
        _chn = 0

        for _shank, _coord in self.coord.items():
            for _channel in _coord:
                _chn += 1
                _data.append({'chn':_chn, 'X':_channel[0], 'Y':_channel[1], 'Z':_channel[2]})
                
        self._sheet = pd.DataFrame(_data, columns=['chn', 'X', 'Y', 'Z'])
        self._sheet.to_csv(os.path.join(to, filename), index=False)
        
    def mni2tal(self, to=None, filename=None):
        
        if isinstance(self._sheet, type(None)):
            self.export_csv()
            
        if filename == None:
            filename = self.patient_id + '_tal_coord.csv'
        
        if to == None:
            to = self._image_dir
        _tal_path = os.path.join(to, filename)    
            
        mni = self._sheet[['X', 'Y', 'Z']].values
    
        rotn = np.array([[1,0,0,0],
                         [0,0.9988,0.05,0],
                         [0,-0.05,0.9988,0],
                         [0,0,0,1]])
        upz = np.array([[0.99,0,0,0],
                        [0,0.97,0,0],
                        [0,0,0.92,0],
                        [0,0,0,1]])
        downz = np.array([[0.99,0,0,0],
                          [0,0.97,0,0],
                          [0,0,0.84,0],
                          [0,0,0,1]])

        inpoints = np.hstack((mni, np.ones((np.size(mni,0), 1)))).T
        _post = inpoints[2, :] < 0

        inpoints[:,  _post] = np.matmul(np.matmul(rotn, downz),inpoints[:,  _post])
        inpoints[:, ~_post] = np.matmul(np.matmul(rotn, upz),  inpoints[:, ~_post])

        _tal_coord = inpoints[:3, :].T
        chn = self._sheet['chn'].values
        self._tal = pd.DataFrame(data= {'x':_tal_coord[:, 0], 'y':_tal_coord[:, 1], 'z':_tal_coord[:, 2]}, columns=['x','y','z'])
        
        self._tal.to_csv(_tal_path, index=False, header=False)
        
        
    def query(self, taljar='/media/STORAGE/EEG/talairach.jar', _from=None, filename=None):

        if filename == None:
            filename = self.patient_id + '_tal_coord.csv'

        if _from == None:
            _from = self._image_dir
            
        _tal_path = os.path.join(_from, filename)
        _tal_labels_path = os.path.join(_from, filename+'.td')

        os.system('java -cp %s org.talairach.ExcelToTD 3:3,%s'%(taljar, _tal_path))

        self.tal_labels = pd.read_csv(_tal_labels_path, header=-1, sep='\t')    
        self.tal_labels.columns = ['x','y','z','hit','Level 1','Level 2','Level 3','Level 4','Level 5']

        _chn = np.zeros(self.tal_labels.x.values.shape, dtype='int')
        _chi = 0
        _previous = [0,0,0]

        for i in range(len(self.tal_labels)):
            if _previous != list(self.tal_labels.loc[i][['x', 'y', 'z']].values):
                _chi += 1
                _previous = list(self.tal_labels.loc[i][['x', 'y', 'z']].values)
            _chn[i] = _chi

        _export_filename = self.patient_id + '_layout.csv'
        self.tal_labels = pd.DataFrame(_chn, columns=['chn']).join(self.tal_labels)
        self.tal_labels.to_csv(os.path.join(_from,_export_filename), index=False)

        return self.tal_labels
