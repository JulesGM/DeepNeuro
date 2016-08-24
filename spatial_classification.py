# Compatibility
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

# Stdlib
import os
import sys
import random
import time

# Own
import utils
import utils.data_utils
import NN_models

# External
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
import mne
import joblib

base_path = os.path.dirname(__file__)

try:
    from numba import jit
except ImportError:
    # Passthrough decorator
    jit = lambda x: x


@jit
def _aux_interp(sample_idx, x, sample_set_idx, sample_bound, fft_bound, sensor_positions, grid, method, picks, show, interp_x):
    if sample_idx % 5 == 0:
        sys.stdout.write("\r\t- cv set: '{}', sample: {} of {} ({:4.2f} %))".format(
            sample_set_idx, sample_idx, sample_bound, 100 * sample_idx / sample_bound))
        sys.stdout.flush()
    for fft_channel_idx in xrange(fft_bound):
        psd_image = scipy.interpolate.griddata(
            sensor_positions, x[sample_set_idx][sample_idx, fft_channel_idx, picks], grid, method, 0)
        interp_x[sample_set_idx][sample_idx, fft_channel_idx] = psd_image


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

    parr = True

    with joblib.Parallel(n_jobs=32,
                         backend="threading") as pool:
        for cv_set in range(2):
            interp_x[cv_set] = np.empty([x[cv_set].shape[utils.X_Dims.samples_and_times.value],
                                         x[cv_set].shape[utils.X_Dims.fft_ch.value], res[0], res[1]],
                                         dtype=np.float32)

            sample_bound = x[cv_set].shape[utils.X_Dims.samples_and_times.value]
            fft_bound = x[cv_set].shape[utils.X_Dims.fft_ch.value]

            if parr:
                pool(joblib.delayed(_aux_interp)(
                        sample_idx=sample_idx, x=x,
                        sample_set_idx=cv_set, sample_bound=sample_bound, fft_bound=fft_bound,
                        sensor_positions=sensor_positions, grid=grid, method=method, picks=picks,
                        show=show, interp_x=interp_x) for sample_idx in xrange(sample_bound))
            else:
                start = time.time()
                for sample_idx in xrange(sample_bound):
                    _aux_interp(
                        sample_idx=sample_idx, x=x,
                        sample_set_idx=cv_set, sample_bound=sample_bound, fft_bound=fft_bound,
                        sensor_positions=sensor_positions, grid=grid, method=method, picks=picks,
                        show=show, interp_x=interp_x
                        )

                    if sample_idx % 1000 == 0 and sample_idx != 0:
                        sys.stderr.write("\ntook {} s for 1000\n".format(time.time() - start))
                        sys.stderr.flush()
                        start = time.time()

            assert np.all(np.isfinite(interp_x[cv_set]))
            interp_x[cv_set] = np.swapaxes(np.swapaxes(interp_x[cv_set], 1, 2), 2, 3)
            print(interp_x[cv_set].shape)
    return interp_x


def _make_image_save_name(res, sensor_type, nfft, fmax, tincr, use_established_bands):
    # we right them all, on purpose, instead of using *args, to make sure everything is in its place
    args = ["_".join([str(d) for d in res]), sensor_type, nfft, fmax, tincr, use_established_bands]
    return "_".join([str(x) for x in args])


def spatial_classification(x, y, nfft, tincr, fmax, info, established_bands,  res, sensor_type,  net_type,
                           learning_rate, minibatch_size, dropout_keep_prob, depth, filter_scale_factor, dry_run, test_qty):

    saves_loc = os.path.join(base_path, "saves/interp_image_saves")
    if not os.path.exists(saves_loc):
        os.mkdir(saves_loc)

    image_name = _make_image_save_name(res, sensor_type, nfft, tincr, fmax, established_bands)
    image_save_name = image_name + ".pkl"
    saver_loader = utils.data_utils.SaverLoader(os.path.join(saves_loc, image_save_name))

    if saver_loader.save_exists():
        prepared_x = saver_loader.load_ds()
    else:
        start = time.time()
        prepared_x = make_interpolated_data(x, res, "cubic", info)
        sys.stderr.write("\n\n")
        print("\nPreparation of the images took {} seconds".format(time.time() - start))
        print("Shape of the prepared dataset {}".format(" & ".join([str(prep_x.shape) for prep_x in prepared_x[:2]])))
        saver_loader.save_ds(prepared_x)

    for i in xrange(2):
        y[i] = utils.to_one_hot(y[i], np.max(y[i]) + 1)

    training_prepared_x = prepared_x[0]
    training_y = y[0]

    validation_prepared_x = prepared_x[1]
    validation_y = y[1]

    x_shape = training_prepared_x.shape
    y_shape_1 = training_y.shape[1]

    if net_type == "vgg":
        print("vgg - not really vgg")
        print("x_shape: {}".format(x_shape))
        summary_path = os.path.join(base_path, "saves", "tf_summaries", "vgg_" + image_save_name + "_" +
                                    str(random.randint(0, 1000000000000)))
        model = NN_models.VGG(x_shape, y_shape_1, dropout_keep_prob=dropout_keep_prob,
                              summary_writing_path=summary_path, expected_minibatch_size=minibatch_size)
        if not dry_run:
            model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                      n_epochs=10000000, minibatch_size=minibatch_size, learning_rate=learning_rate, test_qty=test_qty)
        else:
            print(">>>>> NO FITTING, WAS A DRY RUN")
    elif net_type == "cnn":
        print("cnn")

        summary_path = os.path.join(base_path, "saves", "tf_summaries", "cnn_" + image_name + "_" +
                                    str(random.randint(0, 1000000000000)))
        model = NN_models.CNN(x_shape, y_shape_1, depth=depth, dropout_keep_prob=dropout_keep_prob,
                              filter_scale_factor=filter_scale_factor, summary_writing_path=summary_path,
                              expected_minibatch_size=minibatch_size)
        if not dry_run:

            model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y, n_epochs=10000000,
                      minibatch_size=minibatch_size, learning_rate=learning_rate, test_qty=test_qty)
        else:
            print(">>>>> NO FITTING, WAS A DRY RUN")
    elif net_type == "resnet":
        print("resnet")
        summary_path = os.path.join(base_path, "saves", "tf_summaries", "resnet_" + image_save_name + "_" +
                                    str(random.randint(0, 1000000000000)))
        model = NN_models.ResNet(x_shape, y_shape_1, dropout_keep_prob=dropout_keep_prob,
                                 summary_writing_path=summary_path,
                                 expected_minibatch_size=minibatch_size)
        if not dry_run:
            model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                      n_epochs=10000000, minibatch_size=minibatch_size, learning_rate=learning_rate, test_qty=test_qty)
        else:
            print(">>>>> NO FITTING, WAS A DRY RUN")


    else:
        raise RuntimeError("Received unexpected value '{}' for option --net_type".format(net_type))