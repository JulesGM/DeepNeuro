from __future__ import print_function, generators, division, with_statement
from six import iteritems
from six.moves import zip as izip, range as xrange
from six.moves import cPickle as pickle


# Stdlib
import os
import sys
import glob
import warnings
import logging
import json
import time
import copy

# Own
import utils

# External
import numpy as np
import joblib
import h5py
import mne
import mne.io.pick

from mne.channels.layout import _merge_grad_data
mne.set_log_level("ERROR")

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class SaverLoader(object):
    def __init__(self, path):
        self._save_path = path

    def save_ds(self, data):
        joblib.dump(data, self._save_path, protocol=pickle.HIGHEST_PROTOCOL)

    def load_ds(self):
        return joblib.load(self._save_path,)

    def save_exists(self):
        return os.path.exists(self._save_path)


class HDF5SaverLoader(object):
    def __init__(self, path):
        self._save_path = path

    def load_ds(self):
        """
        The data is saved under names for each of the cross validation sets.
        """
        f = h5py.File(self._save_path, "r")
        new_x = [None for _ in f]
        for k, dataset in f.values():
            new_x[int(k)] = dataset

        return new_x

    def save_ds(self, data, names):
        f = h5py.File(self._save_path, mode="w", libver="latest")
        new_x = []

        for i, cv_set in enumerate(data):
            new_x.append(f.create_dataset(str(i), data=data))

        return new_x

    def save_exists(self):
        return os.path.exists(self._save_path)


def data_gen(base_path, limit = None):
    """
    Generator
    Yields raw files
        -- name, raw, label, len(full_glob)
    """

    base_path = os.path.abspath(base_path)
    assert os.path.exists(base_path), "{base_path} doesn't exist".format(base_path=base_path)
    full_glob = glob.glob(base_path + "/*.fif")
    print("Datagen found {} files".format(len(full_glob)))

    if len(full_glob) == 0:
        raise RuntimeError("Datagen didn't find find any '.fif' files")

    if limit is not None:
        print(">>>>>>>>>> Warning: data_gen limit argument is not None.\n"
              ">>>>>>>>>> This has the effect that only a limited amount ({})\n"
              ">>>>>>>>>> of the data will be loaded. \n".format(limit))

    fif_paths = full_glob[:limit] if limit is not None else full_glob

    if len(fif_paths) == 0:
        raise RuntimeError("fif_path is of size zero.")

    for fif_path in fif_paths:
        name = fif_path.split("/")[-1] # os.path.split appears broken somehow

        # MNE generates a lot of really unnecessary blabla.
        # We realize hiding warnings is far from ideal; we might look into fixing this
        # later on.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = mne.io.Raw(fif_path)
            except ValueError as err:
                print("-- data_gen ValueError:")
                print("-- %s" % name)
                print("-- %s\n" % err)
                raise err
            except TypeError as err:
                print("-- data_gen TypeError")
                print("-- %s" % name)
                print("-- %s\n" % err)
                raise err

        assert name.lower().startswith("k") or name.lower().startswith("r"), \
            "file name is weird, can't guess label from it. ({})".format(name)

        label = name.lower().startswith("k")

        yield name, raw, label, len(full_glob)


BANDS = [
     (0,  4,   'Delta'),
     (4,  8,   'Theta'),
     (8,  12,  'Alpha'),
     (12, 30,  'Beta'),
     (30, 100, 'Gamma')]

LINEAR_HALF_BANDS = [
    (0,  2,   'Delta_0'),
    (2,  4,   'Delta_1'),
    (4,  6,   'Theta_0'),
    (6,  8,   'Theta_1'),
    (8,  10,  'Alpha_0'),
    (10, 12,  'Alpha_1'),
    (12, 21,  'Beta_0'),
    (21, 30,  'Beta_1'),
    (30, 65,  'Gamma_0'),
    (65, 100, 'Gamma_1')]

LINEAR_QUARTER_BANDS = [
    (0,    1,    'Delta_0'),
    (1,    2,    'Delta_1'),
    (2,    3,    'Delta_2'),
    (3,    4,    'Delta_3'),
    (4,    5,    'Theta_0'),
    (5,    6,    'Theta_1'),
    (6,    7,    'Theta_2'),
    (7,    8,    'Theta_3'),
    (8,    9,    'Alpha_0'),
    (9,    10,   'Alpha_1'),
    (10,   11,   'Alpha_2'),
    (11,   12,   'Alpha_3'),
    (12,   16.5, 'Beta_0'),
    (16.5, 21,   'Beta_1'),
    (21,   25.5, 'Beta_2'),
    (25.5, 30,   'Beta_3'),
    (30,   47.5, 'Gamma_0'),
    (47.5, 65,   'Gamma_1'),
    (65,   82.5, 'Gamma_2'),
    (82.5, 100,  'Gamma_3')]

def established_bands(psds, freqs, bands=BANDS):
    assert np.all(np.mean(psds) < 1E6), "We need the raw psds, not the psds converted to dB."
    data = np.empty(shape=(psds.shape[0], len(bands)), dtype=np.float32)

    for i, (fmin, fmax, title) in enumerate(bands):
        freq_mask = (fmin <= freqs) & (freqs < fmax)
        if freq_mask.sum() == 0:
            raise RuntimeError('No frequencies in band "{name}" ({fmin}, {fmax}).\nFreqs:\n{freqs}'.format(
                name=title, fmin=fmin, fmax=fmax, freqs=freqs))

        data[:, i] = np.mean(psds[:, freq_mask], axis=1)

    return data

def maybe_prep_psds(args):
    if args.tmin != 0:
        print("Warning: --tmin is not equal to zero, this is weid. Value : {}".format(args.tmin))
    assert args.tmax >= 40, "tmax is smaller than 40, it's unlikely that that was intended"

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################

    x = [None, None, None] # Generated PSDs
    y = [[], [], []] # Generated labels

    json_path = os.path.join(base_path, "fif_split.json")

    with open(json_path, "r") as json_f:
        fif_split = json.load(json_f) # dict with colors

    split_idx_to_name = {"training":  0,
                         "valid":     1,
                         "test":      2,
                         }

    # Empty folders aren't saved by git
    saves_folder = os.path.join(base_path, "saves")
    if not os.path.exists(saves_folder):
        os.mkdir(saves_folder)
    psd_saves_path = os.path.join(saves_folder, "ds_transform_saves")
    if not os.path.exists(psd_saves_path):
        os.mkdir(psd_saves_path)

    # We build savepaths from different values of the parameters
    saver_loader = SaverLoader(os.path.join(psd_saves_path, "{eb}_{fmax}_{limit}_{tincr}_{nfft}_latest_save.pkl" \
                              .format(eb=args.established_bands, fmax=args.fmax, limit=args.limit,
                                      tincr=args.tincr, nfft=args.nfft)))

    if saver_loader.save_exists():
        print("Loading pickled dataset")
        x, y, info = saver_loader.load_ds()
        print("--")
    else:
        print("Generating the dataset from the raw (.fif) files")
        list_x = [[], [], []]

        shape = None

        for i, (name, raw, label, total) in enumerate(data_gen(args.data_path, args.limit)):
            start_t = time.time()
            files_lim = total if args.limit is None or total > args.limit else args.limit

            split_idx = split_idx_to_name[fif_split[name]]

            lower_bound = args.tmin * 1000
            increment = int(args.tincr * 1000)
            # we ignore args.tmax if it's later than the end of the measure
            delta = min(raw.n_times, args.tmax * 1000) - lower_bound
            # The upper bound is the largest number of samples that allows for complete non overlapping PSD evaluation windows
            # It's the total number of samples minus the rest of the division of the total number of samples by the PSD
            # evaluation window width (the mod).
            # If building something takes exactly 10 minutes, you have 45 minutes but only want full things to be built,
            # then you will be done in
            # 45 - 45 % 10 = 45 - 5 = 40 minutes
            # if you start at 0h10,
            # you will be done at
            # 0h10 + 45 - 45 % 10 = 0h50 (this last part is pretty obvious)
            upper_bound = lower_bound + delta - delta % int(args.tincr * 1000)

            for j, psd_band_t_start_ms in enumerate(xrange(lower_bound, upper_bound, increment)):
                """
                So, GLOB_* are in seconds, and raw.n_times is in milliseconds.
                Iterating on a range requires ints, and we don't want to lose precision by rounding up the milliseconds to seconds,
                so we put everything in milliseconds.

                mne.time_frequency.psd_welch takes times in seconds, but accepts floats, so we divide by 1000 while
                keeping our precision.
                """
                if j % 10 == 0:
                    sys.stderr.write("\r\t- File {current} of {files_lim} (max: {max_possible}) - Segment {seg_start:7}"
                                     " of {seg_end:7}, {percentage:<4.2f}%".format(
                                        current=i + 1, files_lim=files_lim, max_possible=total,
                                        seg_start=str(psd_band_t_start_ms), seg_end=str(upper_bound),
                                        percentage=100 * psd_band_t_start_ms / upper_bound))
                    sys.stderr.flush()

                psds, freqs = mne.time_frequency.psd_welch(n_jobs=1, # in our tests, more jobs invariably resulted in slower execution, even on the 32 cores xeons of the Helios cluster.
                                     inst=raw,
                                     picks=mne.pick_types(raw.info, meg=True),
                                     n_fft=args.nfft,
                                     n_overlap=args.noverlap,
                                     tmin=psd_band_t_start_ms / 1000.,
                                     tmax=psd_band_t_start_ms / 1000. + args.tincr,
                                     fmax=(min(100, args.fmax) if args.established_bands else args.fmax),
                                     verbose="INFO"
                                     )

                if args.sensor_type == "grad":
                    psds = _merge_grad_data(psds)

                if args.established_bands:
                    if args.established_bands == "half":
                        psds = established_bands(psds, freqs, LINEAR_HALF_BANDS)
                    elif args.established_bands == "quarter":
                        psds = established_bands(psds, freqs, LINEAR_QUARTER_BANDS)
                    elif args.established_bands == True:
                        psds = established_bands(psds, freqs, BANDS)
                    else:
                        raise ValueError("Unsupported value for argument established_bands: {}".format(
                            args.established_bands))

                num_res_db = 10 * np.log10(psds)

                if not np.all(np.isfinite(num_res_db)):
                    sys.stderr.write("\n>>>>>>>>> {} : has a NAN or INF or NINF post log - skipping this segment ({}:{})\n" \
                          .format(name, psd_band_t_start_ms, upper_bound))
                    sys.stderr.flush()
                    continue


                if shape is None:
                    print(num_res_db.shape)

                if shape != num_res_db.shape:
                    print("from {} to {}".format(shape, num_res_db.shape))
                    shape = num_res_db.shape

                list_x[split_idx].append(num_res_db)
                y[split_idx].append(label)

            sys.stderr.write("\ntime: {} s\n".format(time.time() - start_t))

        assert len(list_x) == 3
        assert len(y) == 3

        # Make sure we have samples in each of the cross validation sets
        x_lens = [len(_x) for _x in list_x]
        for i, x_len in enumerate(x_lens):
            print((i, x_len))
            assert x_len > 0, "cross validation set #{} of x is empty. ".format(i)

        for i in xrange(3):
            x[i] = np.dstack(list_x[i])
            # We convert the PSD list of ndarrays to a single multidimensional ndarray
            x[i] = x[i].astype(np.float32)
            # We do the same with the labels
            y[i] = np.asarray(y[i], np.float32)
            # Transpose for convenience
            x[i] = x[i].T

            assert len(x[i].shape) == utils.X_Dims.size.value
            assert x[i].shape[utils.X_Dims.samples_and_times.value] == y[i].shape[0], x[i].shape[utils.X_Dims.samples_and_times.value]  # no_samples

            assert np.all(np.isfinite(x[i]))

        # Take any valid file's position information, as all raws [are supposed to] have the same positions.
        # Deep copying it allows the garbage collector to release the raw file. Not major at all.. but still.
        info = copy.deepcopy(next(data_gen(args.data_path))[1].info)
        print("--")
        print("Saving the newly generated dataset")
        saver_loader.save_ds((x, y, info))
        print("--")



    return x, y, info