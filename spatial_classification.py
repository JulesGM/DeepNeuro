# Compatibility
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

# Stdlib
import os
import sys

# Own
import utils
import utils.data_utils
import NN_models

# External
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
import mne

base_path = os.path.dirname(__file__)


def make_interpolated_data(x, res, method, sample_info, sensor_type=True, show=False):
    picks = mne.pick_types(sample_info, meg=sensor_type)
    sensor_positions = mne.channels.layout._auto_topomap_coords(sample_info, picks, True)
    # Take any valid file's position information, as all raws [are supposed to] have the same positions
    assert len(sensor_positions.shape) == 2 and sensor_positions.shape[1] == 2, sensor_positions.shape[1]
    min_x = np.floor(np.min(sensor_positions[:, 0]))
    max_x = np.ceil(np.max(sensor_positions[:, 0]))
    min_y = np.floor(np.min(sensor_positions[:, 1]))
    max_y = np.ceil(np.max(sensor_positions[:, 1]))

    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, res[0],), np.linspace(min_y, max_y, res[1]))
    grid = (grid_x, grid_y)

    interp_x = [None, None, None]

    # Merge grad data
    ## picks = mne.channels.layout._pair_grad_sensors(sample_info)
    ## x = mne.channels.layout._merge_grad_data(x[picks])

    for sample_set_idx in xrange(2): # There is currently no reason to do the test set. it being hardcoded is not ideal,
                                     # but such is life
        interp_x[sample_set_idx] = np.empty([x[sample_set_idx].shape[utils.X_Dims.samples_and_times.value],
                                             x[sample_set_idx].shape[utils.X_Dims.fft_ch.value], res[0], res[1]],
                                            dtype=np.float32)
        sample_bound = x[sample_set_idx].shape[utils.X_Dims.samples_and_times.value]
        fft_bound = x[sample_set_idx].shape[utils.X_Dims.fft_ch.value]

        for sample_idx_per_set in xrange(sample_bound):
            if sample_idx_per_set % 5 == 0:
                sys.stdout.write("\r\t- cv set: '{}', sample: {} of {} ({:4.2f} %))".format(
                    sample_set_idx, sample_idx_per_set, sample_bound, 100 * sample_idx_per_set / sample_bound))
                sys.stdout.flush()
            for fft_channel_idx in xrange(fft_bound):
                psd_image = scipy.interpolate.griddata(
                    sensor_positions, x[sample_set_idx][sample_idx_per_set, fft_channel_idx, picks], grid, method, 0)
                interp_x[sample_set_idx][sample_idx_per_set, fft_channel_idx] = psd_image

                if show:
                    plt.imshow(psd_image, interpolation="none")
                    plt.show()

        assert np.all(np.isfinite(interp_x[sample_set_idx]))
        interp_x[sample_set_idx] = np.swapaxes(np.swapaxes(interp_x[sample_set_idx], 1, 2), 2, 3)

    return interp_x


def make_image_save_name(res, sensor_type, nfft, fmax, tincr, use_established_bands):
    # we right them all, on purpose, instead of using *args, to make sure everything is in its place
    args = [res, sensor_type, nfft, fmax, tincr, use_established_bands]
    return "_".join([str(x) for x in args]) + ".pkl"


def spatial_classification(x, y,  nfft, tincr, fmax, info, established_bands,  res, sensor_type,  net_type,
                           learning_rate, minibatch_size, dropout_keep_prob, depth, filter_scale_factor):

    saves_loc = os.path.join(base_path, "saves/interp_image_saves")
    if not os.path.exists(saves_loc):
        os.mkdir(saves_loc)

    image_save_name = make_image_save_name(res, sensor_type, nfft, tincr, fmax, established_bands)
    saver_loader = utils.data_utils.SaverLoader(os.path.join(saves_loc, image_save_name))

    if saver_loader.save_exists():
        prepared_x = saver_loader.load_ds()
    else:
        print("Preparing the images")
        prepared_x = make_interpolated_data(x, res, "cubic", info)
        saver_loader.save_ds(prepared_x)

    for i in xrange(2):
        y[i] = utils.to_one_hot(y[i], np.max(y[i]) + 1)

    training_prepared_x = prepared_x[0]
    training_y = y[0]

    validation_prepared_x = prepared_x[1]
    validation_y = y[1]

    x_shape = training_prepared_x.shape
    y_shape_1 = training_y.shape[1]

    if net_type == "cnn":
        print("cnn")
        model = NN_models.CNN(x_shape, y_shape_1, depth=depth, dropout_keep_prob=dropout_keep_prob,
                              filter_scale_factor=filter_scale_factor)
        model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y, n_epochs=10000000,
                              minibatch_size=minibatch_size, learning_rate=learning_rate)

    elif net_type == "resnet":
        print("resnet")
        model = NN_models.ResNet(x_shape, y_shape_1, depth=depth, dropout_keep_prob=dropout_keep_prob,
                                 filter_scale_factor=filter_scale_factor)
        model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                  n_epochs=10000000, minibatch_size=minibatch_size, learning_rate=learning_rate)
    else:
        raise RuntimeError("Received unexpected value '{}' for option --net_type".format(net_type))