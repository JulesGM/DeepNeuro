from __future__ import division, print_function, with_statement
range = xrange
import matplotlib as mpl
import tensorflow as tf

import sys, os, re, fnmatch, subprocess as sp, argparse as ap, logging
from collections import defaultdict
import mne
import mne.io.pick
import mne.viz as viz

from data_utils import *
from utils import *




def topomap_array(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
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
    return Xi, Yi, Zi

def main(argv):
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("The bas_path is '%s'" % base_path)


    defaults = dict(
        data_path ="/media/jules/JULES700GB/Dropbox/aut_gamma/MEG",
        hdf5_path = "/media/jules/JULES700GB/COCOLAB/data.hdf5"
    )

    mne.set_log_file("./mne_log.log", overwrite=True)

    # args = parse_args(argv, defaults)

    target_name = "K0002_rest_raw_tsss_mc_trans_ec_ica_raw.fif"
    target = os.path.join(base_path, target_name)
    raw = Raw(target)

    mag_idx = pick_types(raw.info, meg="mag")
    mag_chs = np.array([raw.info["chs"][idx] for idx in mag_idx])
    pos_mag = get_pos(mag_chs, "topomap")

    data, times = raw[mag_idx, :]

    """
    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
               'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
               'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
    """

    mpl.use('Agg')
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    anims = []

    if len(argv) == 3:
        proc_no = argv[1] - 1
        interval = argv[2]
        print(">> Multi_proc :: proc %s started" % proc_no)

    else:
        proc_no = None
        interval = 100

    def work(j, interval):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print((j, j + interval))
        for i in range(j, j + interval):
            if i % 100 == 0 and i != 0:
                print(i)
            im, cn = viz.plot_topomap(data[:, i], pos_mag, axes=ax, show=False,
                                      image_interp="bicubic", sensors=True, contours=False,
                                      outlines=None)
            anims.append([im])

        anim = animation.ArtistAnimation(fig, anims, interval=333, repeat_delay=0)
        #ffwriter = animation.FFMpegWriter()
        path = os.path.join(base_path, "output", "vid%s.mp4" % str(j))
        print("TRYING TO SAVE TO '%s'" % path)
        anim.save(path, fps=30, dpi=500,
                  #writer=ffwriter,
                  #extra_args=['-vcodec', 'libx264']
                  )

    if proc_no is None:
        for j in range(0, 1000, interval):
            work(j, interval)
    else:
        work(proc_no * interval, interval)

    return 0


if __name__ == "__main__": sys.exit(main(sys.argv))