
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

