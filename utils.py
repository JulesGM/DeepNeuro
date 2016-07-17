from __future__ import print_function, division, with_statement

from mne.io import Raw, Info
from mne.io.pick import channel_type, channel_indices_by_type, pick_types
from mne.channels.layout import _auto_topomap_coords
from mne.channels.channels import _contains_ch_type
import numpy as np

from _ext_modules import *
#/usr/local/lib/python2.7/dist-packages/mne


def azim_proj(pos):
    """
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


def positions_from_channels(channels):
    """
    :param channels:
    :return: position
    """
    return np.array([ch["loc"][:] for ch in channels])


def get_pos(channels, kind="topomap"):
    """
    Return positions from channels.
    :param channels:
    :param kind: one of "topomap", "azim_proj", "2d", "3d".
        "topomap" and "azim_proj" are the same.
    :return: positions
    """
    all_pos = positions_from_channels(channels)
    if kind == "topomap" or "azim_proj":
        pos_3d = all_pos[:, :3]
        pos = azim_proj(pos_3d)
        #pos = _auto_topomap_coords(info, picks, True)
    elif kind == "2d":
        pos = all_pos[:, :2]
        # we hate using a private function, but it works for the moment
    elif kind == "3d":
        pos = all_pos[:, :3]
    else:
        raise RuntimeError("should never happen")
    return pos


def get_deg_pos(pos):
    """
    Converts to radial positions.
    :param pos:
    :return:
    """

    dims = pos.shape[1]
    if dims == 2:
        return cart_to_polar(pos.astype(np.float32))
        # radial coords
    elif dims == 3:
        # spherical coords
        return cart_to_spherical(pos.astype(np.float32))
    else:
        raise RuntimeError("positions need to be of size 2 or three, "
                           "received size %s" % dims)


def own_plot_sensors(dots, interpolate = False):
    from matplotlib import pyplot as plt

    if interpolate:

        methods = [None, 'none', 'nearest', 'bilinear', 'bicubic',
                   'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
                   'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel',
                   'mitchell', 'sinc', 'lanczos']

        fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                                 subplot_kw={'xticks': [], 'yticks': []})

        for ax, interp_method in zip(axes.flat, methods):
            ax.imshow(dots, interpolation=interp_method)
            ax.set_title(interp_method)

    if dots.shape[1] == 2:
        fig, ax = plt.subplots()
        ax.plot(dots[:, 0], dots[:, 1], "o")
        ax.grid(True)

    elif dots.shape[1] == 3:
        import mpl_toolkits.mplot3d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2], picker=True)

        ax.azim = 90
        ax.elev = 90


        ax.grid(True)

    plt.show()
