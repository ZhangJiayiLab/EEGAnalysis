import numpy as np

def get_relative_behavior_time(marker, behavior):
    '''get_relative_behavior_time
    compare the behavior time and the marker time,
    find the closest behavior relative time for each marker.
    
    arguments:
    - marker:   numpy.ndarray
    - behavior: numpy.ndarray
    
    return:
    - relative_time: numpy.ndarray
    '''

    _rel = {}
    for idx, each in enumerate(behavior):
        _target = np.argmin(np.abs(marker - each))
        _delta = each - marker[_target]

        if _delta > 4:  #XXX: there are some strange points!
            continue

        if _target not in _rel.keys():
            _rel[_target] = [(idx, _delta)]
        else:
            _rel[_target].append((idx, _delta))

    _result = np.zeros_like(marker)*np.nan
    for item, val in _rel.items():
        if len(val) == 1:
            _result[item] = val[0][1]
        else:
            _queue = np.array([i[1] for i in val])
            _min = np.max(_queue < 0) if len(_queue < 0) > 0 else np.min(_queue)
            _result[item] = _min

    return _result