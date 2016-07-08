from six.moves import cPickle as pickle
import sys, os, re, fnmatch, subprocess as sp, argparse as ap, logging, threading
import os, fnmatch, argparse as ap

import numpy as np

from mne.io import Raw
import mne.viz as viz
import mne.io.pick

from utils import *

from matplotlib import use
#use('Agg'); del use
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# This is assumed to not change.
TOPOMAP_DEFAULTS = dict(res=64, sensors=True, contours=None, image_interp="none",)


def topomap_array(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=128, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head', image_mask=None,
                 contours=6, image_interp='bicubic', show=True,
                 head_pos=None, onselect=None, axis=None):
    """
   Taken from mne.viz.plot_topomap
   :return:
    """
    from mne.io.pick import (pick_types, _picks_by_type, channel_type, pick_info,
                           _pick_data_channels)
    from mne.utils import _clean_names, _time_mask, verbose, logger, warn
    from mne.viz.utils import (tight_layout, _setup_vmin_vmax, _prepare_trellis,
                        _check_delayed_ssp, _draw_proj_checkbox, figure_nobar,
                        plt_show, _process_times)
    from mne.channels.layout import _find_topomap_coords
    from mne.viz.topomap import _check_outlines, _prepare_topomap, _griddata
    from matplotlib import pyplot as plt

    data = np.asarray(data)

    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos)  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = set(channel_type(pos, idx)
                      for idx, _ in enumerate(pos["chs"]))
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.channels.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object and "
                             "the data array does not match. " + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            from mne.channels.layout import (_merge_grad_data, find_layout,
                                           _pair_grad_sensors)
            picks, pos = _pair_grad_sensors(pos, find_layout(pos))
            data = _merge_grad_data(data[picks]).reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks)

    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)

    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    pos, outlines = _check_outlines(pos, outlines, head_pos)

    ax = axes if axes else plt.gca()
    pos_x, pos_y = _prepare_topomap(pos, ax)
    if outlines is None:
        xmin, xmax = pos_x.min(), pos_x.max()
        ymin, ymax = pos_y.min(), pos_y.max()
    else:
        xlim = np.inf, -np.inf,
        ylim = np.inf, -np.inf,
        mask_ = np.c_[outlines['mask_pos']]
        xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                      np.max(np.r_[xlim[1], mask_[:, 0]]))
        ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                      np.max(np.r_[ylim[1], mask_[:, 1]]))

    # interpolate data
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _griddata(pos_x, pos_y, data, Xi, Yi)

    return Zi


def as_array(raw, meg_type, starting_frame, interval,
                   topomap_kwargs):
    """
    Returns the numerical array used by viz.plot_topomap to generate its
    plots. Calls topomap_array, a copy of the logic used by viz.plot_topomap.
    ------------
    :return:
    frames: the frames of the segment of the raw object we wanted the topomaps of.

    """

    frames = []
    idx = pick_types(raw.info, meg=meg_type)
    chs = np.array([raw.info["chs"][i] for i in idx])
    pos = get_pos(chs, "topomap")
    data, times = raw[idx, :]
    j = starting_frame
    print((j, j + interval))

    for i in range(j, j + interval):
        if i % 100 == 0 and i != 0:
            print(i)

        array = topomap_array(data[:, i], pos, show=False,
                              **topomap_kwargs)
        frames.append(array)

    return frames


def as_image_array(raw, meg_type, starting_frame, interval,
                   plot_kwags):
    """
    Returns viz.plot_topomap images called on a segment of the raw object's samples,
    the segment being (starting_frame, starting_frame + interval).
    ------------
    :param raw: mne raw object
    :param meg_type: can be True for both mag and grad, or "mag", or "grad"
    :param starting_frame: the frame in raw's mesures we need to begin at
    :param interval: length of the data segments
    :param image_interp: Matplotlib interpolation method
    All the possible methods of matplotlib interpolations are acceptable:
    [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                   'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
                   'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    ------------
    :return:
    frames: the frames of the piece of anim
    fig: the matlab figure with which the topomap was generated

    """

    frames = []

    idx = pick_types(raw.info, meg=meg_type)
    chs = np.array([raw.info["chs"][i] for i in idx])
    pos = get_pos(chs, "topomap")
    data, times = raw[idx, :]

    j = starting_frame
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(j, j + interval):
        if i % 100 == 0 and i != 0:
            print(i)

        im, cn = viz.plot_topomap(data[:, i], pos, axes=ax, show=False,
                                  **plot_kwags)
        frames.append([im])
    return frames, fig


def as_anim(frames, fig, save_path, anim_interval,
            fps, dpi):
    """
    Compresses the image frames to an animation format & saves the result to a
    save_path.
    ------------
    :return:
    returns the animation object

    """

    anim = animation.ArtistAnimation(fig, frames, interval=anim_interval,
                                     repeat_delay=0)
    # to make it more clear when a file's compression has been completed,
    # we give it the extension "part" when it's not done
    anim.save(save_path, fps=fps, dpi=dpi)
    print("'%s' should be done." % save_path)
    return anim


def make_anims(raw, meg_type, base_path, j, interval, frames, threads, fps, dpi, topomap_kwargs = TOPOMAP_DEFAULTS):
    """
    Generates video animations with the segments, and saves them individually to
    basepath/output/segmentstart_segmentend.mp4.
    Creates thread objects to do the compressing & saving (one does the two).
    """

    save_path = os.path.join(base_path, "output", "%s_%s.mp4" %
                             (j, j + interval))
    loc_frames, fig = as_image_array(raw, meg_type, j, interval, topomap_kwargs)

    # the function as_anim is native extension bound, so free of too much GIL
    # restriction on parallelism
    if threads is None:
        as_anim(loc_frames, fig, save_path, interval, fps, dpi)

    else:
        th = threading.Thread(target=as_anim, args=(
            loc_frames, fig, save_path, interval, fps, dpi))
        threads.append(th)
        th.start()


def make_arrays(uperrange, raw, proc_no, interval, base_path,
                meg_type, ANIM = False, fps = 10, dpi = 300, topomap_kwargs = TOPOMAP_DEFAULTS):
    """
    Generates the frames for the raw file
    ------------
    :return: The frames, if in array mode
    """

    frames = []
    if proc_no is None:
        threads = []
        for j in range(0, uperrange, interval):
            if ANIM:
                make_anims(raw, meg_type, base_path, j,
                                         interval, frames, threads, fps, dpi)
            else:
                frames.extend(as_array(raw, meg_type, j, interval, topomap_kwargs))

        # Currently, for anims only
        for th in threads:
            th.join()
    else:
        j = proc_no * interval
        if ANIM:
            make_anims(raw, meg_type, base_path, j,
                                     interval, frames, None, fps, dpi)
        else:
            frames.extend(as_array(raw, meg_type, j, interval, topomap_kwargs))

    return np.array(frames)


def parse_args(argv, defaults):
    parser = ap.ArgumentParser()
    parser.add_argument("--data_path", "-d", default=defaults["data_path"])
    parser.add_argument("--hdf5_path", "-H", default=defaults["hdf5_path"])
    return parser.parse_args(argv[1:])

"""
def make_samples(raw, labels, time_slice_length,
                 time_slice_skip=0, sample_time_overlap=None):



    for i, r in enumerate(raw):
        while True:
            assert False, "TODO"
            label = labels[i]
            feature = raw[i][i]
"""

def maybe_load_data(data_path, limit = None):
    print("Loading from fif.")
    raw = []
    labels = []
    filenames = []
    failed = 0
    total = 0

    for sub_folder_name in os.listdir(data_path):
        print(sub_folder_path)
        sub_folder_path = os.path.join(data_path, sub_folder_name)

        if not os.path.isdir(sub_folder_path):
            print("'%s' is not a dir" % sub_folder_path)
            continue

        for file_name in os.listdir(sub_folder_path):
            if fnmatch.fnmatch(file_name, "*.fif") and "timecours" not in file_name:

                print("at %s of %s, %s good" % (total, limit if limit is not None else "probably %s" % 651, total - failed))
                total += 1

                file_path = os.path.join(sub_folder_path, file_name)

                try:
                    entry = Raw(file_path, preload=True)

                except ValueError:
                    # print("- File '%s' had no data. Skipping it." % file_path)
                    failed += 1
                    continue

                # Closer to being ok
                raw.append(entry)
                filenames.append(file_name)
                labels.append(file_name[0].lower() == "k")

                if limit and total - failed > limit:
                    break

        if limit and total - failed > limit:
            break

    print("TOTAL FAILED RATIO: %s" % (failed * 100. / total))

    return raw, np.array(labels)
