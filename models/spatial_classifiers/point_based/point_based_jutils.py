from __future__ import division, absolute_import, print_function
import numpy as np

def make_pointbased(x):
	""" input: x with the dimensions being [samples, fft_channels, sensors]
		output: x with the dimensions being [samples, sensors, fft_channels]
	""" 

	return np.moveaxis(x, 1, 2)


def make_pointbased_with_positions(x):
	beg_shape = x.shape
	x = make_pointbased(x)

    if sensor_type == "grad":
        picks = list(range(x[0].shape[2]))
        fake_picks = pick_types(info, meg="grad")

        no_chs = len(picks)
        sensor_positions = _auto_topomap_coords(info, fake_picks, True)[:no_chs * 2:2]
        assert sensor_positions.shape[0] == no_chs, (sensor_positions.shape, no_chs)
        assert x[0].shape[2] == no_chs, (x[0].shape[2], no_chs)

        # Make sure all positions are unique
        no_positions = sensor_positions.shape[0]
        uniques = np.vstack({tuple(row) for row in sensor_positions})
        assert no_positions == uniques.shape[0]

    else:
        picks = pick_types(info, meg=sensor_type)
        sensor_positions = _auto_topomap_coords(info, picks, ignore_overlap=True)

    assert len(sensor_positions.shape) == 2 and sensor_positions.shape[1] == 2, sensor_positions.shape[1]

    sensor_positions[:, 0] -= np.mean(sensor_positions[:, 0])
    sensor_positions[:, 1] -= np.mean(sensor_positions[:, 1])
    sensor_positions[:, 2] -= np.mean(sensor_positions[:, 2])
    
    sensor_positions[:, 0] /= np.std(sensor_positions[:, 0])
    sensor_positions[:, 1] /= np.std(sensor_positions[:, 1])
    sensor_positions[:, 2] /= np.std(sensor_positions[:, 2])

	pos_ch = np.repeat(sensor_positions.reshape(1, -1, 3), x.shape[0], axis=0)    
	x = np.concatenate([x, pos_ch])
	
	assert x.shape[0] == beg_shape[0], (x.shape[0], beg_shape[0])
	assert x.shape[1] == beg_shape[1], (x.shape[1], beg_shape[1])
	assert x.shape[2] == beg_shape[2] + 3, (x.shape[2], beg_shape[2], beg_shape[2] + 3)

	return x