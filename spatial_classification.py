
#! /usr/bin/env python

# Compatibility imports
from __future__ import with_statement, print_function, division
from six import iteritems
from six.moves import zip as izip

# Stdlib imports
import sys, os, argparse, time, logging, enum, json, subprocess as sp

from utils.data_utils import *

# scipy/numpy/matplotlib/tf
import numpy as np
import tensorflow as tf

from collections import Counter

# MNE imports
import mne, mne.time_frequency
from mne.decoding import CSP
from mne.channels.layout import _auto_topomap_coords


# Varia
import tflearn
import h5py

import utils
from utils.data_utils import SaverLoader

def spatial_classification(interp_X, interp_Y, train_picks, valid_picks, test_picks):
    assert False, "This code is not functional"
    # normalization
    interp_X = (interp_X - np.average(interp_X)) / np.std(interp_X)

    # Real-time data augmentation
    # img_aug = tflearn.ImageAugmentation()
    # img_aug.add_random_flip_leftright()
    # img_aug.add_random_rotation(max_angle=5)

    # img_prep = ImagePreprocessing()
    # img_prep.add_featurewise_zero_center()
    # img_prep.add_featurewise_stdnorm()

    # Convolutional network building
    network = tflearn.input_data(shape=[None, 32, 32, 3],
                         #data_preprocessing=img_prep,
                         #data_augmentation=img_aug
                                 )
    network = tflearn.conv_2d(network, 32, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 10, activation='softmax')
    network = tflearn.regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(interp_X, interp_Y, n_epoch=50, shuffle=True, validation_set=(interp_X[valid_picks], interp_Y[valid_picks]),
              show_metric=True, batch_size=96)

    classifiers = [model]
    one_hot_set = {model}

    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            features_tr = interp_X[train_picks, :]
            labels_tr = to_one_hot(interp_Y[train_picks], 2)

            features_va = interp_X[valid_picks]
            labels_va = to_one_hot(interp_Y[valid_picks], 2)

            classifier.fit(features_tr, labels_tr, n_epoch=10000, validation_set=(features_va, labels_va))

            predicted_va = np.argmax(classifier.predict(features_va), axis=1)

            print(np.mean(predicted_va == labels_va))
            print(Counter(predicted_va))
            print(Counter(interp_Y[valid_picks].tolist()))

        else:
            raise RuntimeError("Landed in a dead section")



def make_interpolated_data(X, res, method, sample_info, sensor_type=True, show=False):
    picks = mne.pick_types(sample_info, meg=sensor_type)
    sensor_positions = _auto_topomap_coords(sample_info, picks, True)

    # Take any valid file's position information, as all raws [are supposed to] have the same positions
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    assert len(sensor_positions.shape) == 2 and sensor_positions.shape[1] == 2, sensor_positions.shape[1]
    min_x = np.floor(np.min(sensor_positions[:, 0]))
    max_x = np.ceil(np.max(sensor_positions[:, 0]))
    min_y = np.floor(np.min(sensor_positions[:, 1]))
    max_y = np.ceil(np.max(sensor_positions[:, 1]))

    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, res[0],), np.linspace(min_y, max_y, res[1]))
    grid = (grid_x, grid_y)

    interp_x = np.empty([X.shape[X_Dims.samples_and_times.value], X.shape[X_Dims.fft_ch.value], res[0], res[1]], dtype=np.float32)

    i_bound = X.shape[X_Dims.samples_and_times.value]
    j_bound = X.shape[X_Dims.fft_ch.value]
    for i in xrange(i_bound):
        if i % 5 == 0:
            sys.stdout.write("\ri: {} of {} ({:4.2f} %))".format(i, i_bound, 100 * i / i_bound))
        for j in xrange(j_bound):
            psd_image = griddata(sensor_positions[picks, :], X[i, j, picks], grid, method)
            interp_x[i, j, :] = psd_image[:, :]

            if show:
                plt.imshow(psd_image, interpolation="none")
                plt.show()

    return interp_x
