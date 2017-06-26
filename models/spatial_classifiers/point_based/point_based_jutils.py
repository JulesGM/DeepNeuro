from __future__ import division, absolute_import, print_function
import numpy as np
import os, sys
from mne.channels.layout import pick_types, _auto_topomap_coords



def make_pointbased(x):
    """ input: x with the dimensions being [samples, fft_channels, sensors]
        output: x with the dimensions being [samples, sensors, fft_channels]
    """ 

    return np.moveaxis(x, 1, 2)
    


def make_pointbased_with_positions(x, info, sensor_type="grad"):
    x = make_pointbased(x)
    beg_shape = x.shape

    if sensor_type == "grad":
        picks = pick_types(info, meg="grad")
        sensor_positions = _auto_topomap_coords(info, picks, ignore_overlap=True) # [:no_chs * 2:2]
    else:
        picks = pick_types(info, meg=sensor_type)
        sensor_positions = _auto_topomap_coords(info, picks, ignore_overlap=True)


    sensor_positions[:, 0] -= np.mean(sensor_positions[:, 0])
    sensor_positions[:, 1] -= np.mean(sensor_positions[:, 1])
    
    sensor_positions[:, 0] /= np.std(sensor_positions[:, 0])
    sensor_positions[:, 1] /= np.std(sensor_positions[:, 1])

    pos_ch = np.repeat(sensor_positions.reshape(1, -1, 2), x.shape[0], axis=0)    
    x = x[:, picks, :]


    x = np.concatenate([x, pos_ch], axis=2)
    print(x.shape)    

    assert x.shape[0] == beg_shape[0]
    assert x.shape[1] == len(picks)
    assert x.shape[2] == beg_shape[2] + 2, (x.shape[2], beg_shape[2] + 2)
    return x