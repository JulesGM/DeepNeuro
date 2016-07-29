from __future__ import print_function, division, with_statement

from mne.io import Raw, Info
from mne.io.pick import channel_type, channel_indices_by_type, pick_types
from mne.channels.layout import _auto_topomap_coords
from mne.channels.channels import _contains_ch_type
import numpy as np

import _ext_modules
from _ext_modules import *
#/usr/local/lib/python2.7/dist-packages/mne

import enum
class X_Dims(enum.Enum):
    samples_and_times = 0
    fft_ch = 1
    sensors = 2
    size = 3


def to_one_hot(input, max_classes):
    no_samples = input.shape[0]
    output = np.zeros((input.shape[0], max_classes), np.float32)
    output[np.arange(no_samples), input.astype(np.int32)] = 1
    return output


def from_one_hot(values):
    return np.argmax(values, axis=1)


def azim_proj(pos):
    """
    This is an implementation of _topomap_coords

    Bashivan, Pouya, et al. "Learning Representations from EEG with Deep
    Recurrent-Convolutional Neural Networks." arXiv preprint arXiv:1511.06448
    (2015).
    From https://github.com/pbashivan/EEGLearn/blob/master/eeg_cnn_lib.py
    :param pos:
    :return:
    """

    elev_az_r = cart_to_spherical(pos.astype(np.float32))
    elev, az, r = elev_az_r [:, 0], elev_az_r [:, 1], elev_az_r [:, 2]

    proj = np.empty((az.shape[0], 2), np.float32)
    proj[:, 0] = az[:]
    proj[:, 1] = np.ones_like(elev) * (np.pi / 2.0) - elev

    return polar_to_cart(proj)