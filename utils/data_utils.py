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
import scipy
import joblib
import h5py
import mne
import mne.io.pick
from mne.channels.layout import pick_types, _auto_topomap_coords

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
        self.save_path = path

    def load_ds(self):
        """
        The data is saved under names for each of the cross validation sets.
        """
        f = h5py.File(self.save_path, "r")
        new_x = [None for _ in f]
        for k, dataset in f.items():
            new_x[int(k)] = dataset

        return new_x[()]

    def save_ds(self, data, names):
        f = h5py.File(self.save_path, mode="w", libver="latest")
        new_x = []

        for i, cv_set in enumerate(data):
            new_x.append(f.create_dataset(str(i), data=data))

        return new_x

    def save_exists(self):
        return os.path.exists(self.save_path)


def data_gen(base_path, limit=None):
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

def _psd_time_unit_cut(args, i, j, raw, files_lim, total, psd_band_t_start_ms, upper_bound, shape, name):
    """ aux function to stratify abstraction levels

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



    psds, freqs = mne.time_frequency.psd_welch(n_jobs=1,
                                               inst=raw,
                                               picks=mne.pick_types(raw.info, meg=args.sensor_type),
                                               n_fft=args.nfft,
                                               n_overlap=args.noverlap,
                                               tmin=psd_band_t_start_ms / 1000.,
                                               tmax=psd_band_t_start_ms / 1000. + args.tincr,
                                               fmax=(min(100, args.fmax) if args.established_bands else args.fmax), # wtf is this
                                               verbose="INFO"
                                               )

    # We don't want to merge grads anymore
    # if args.sensor_type == "grad":
    #    # We already only have grad values
    #    psds = _merge_grad_data(psds)



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
        return

    if shape is None:
        # print(num_res_db.shape)
        pass
    
    elif shape != num_res_db.shape:    
        shape = num_res_db.shape

    return shape, num_res_db


def maybe_prep_psds(args):
    if args.tmin != 0:
        print("Warning: --tmin is not equal to zero, this is weid. Value : {}".format(args.tmin))
    assert args.tmax >= 40, "tmax is smallerplus.google.com than 40, it's unlikely that that was intended"

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################

    x = [None, None, None] # Generated PSDs
    y = [[], [], []] # Generated labels

    json_path = os.path.join(base_path, "fif_split.json")

    with open(json_path, "r") as json_f:
        fif_split = json.load(json_f) # dict with colors

    crossvalidation_set_to_name = {
                         "training":  0,
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
    saver_loader = SaverLoader(os.path.join(psd_saves_path, "{is_time_dependant}_{eb}_{fmax}_{limit}_{tincr}_{nfft}_latest_save.pkl" \
                              .format(is_time_dependant=args.is_time_dependant, eb=args.established_bands,
                                      fmax=args.fmax, limit=args.limit, tincr=args.tincr, nfft=args.nfft)))

    if saver_loader.save_exists():
        print("Loading pickled dataset")
        x, y, info = saver_loader.load_ds()
        print("--")
    else:
        print("Generating the dataset from the raw (.fif) files")
        list_x = [[], [], []]
        shape = None

        for i, (name, raw, label, total) in enumerate(data_gen(args.data_path, args.limit)):



            if name not in fif_split:
                print(">>>>> '{}' is not in the 'fif_split.json' file!!!!".format(name))
                raw_input()
                continue

            files_lim = total if args.limit is None or total > args.limit else args.limit
            crossvalidation_set = crossvalidation_set_to_name[fif_split[name]]

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

            # If we are using time dependant data, we need to keep the samples of each file seperated.
            # Here, we insert an array to contain the samples of a specific file.
            # (We have no reason to save the file name, or anything of the sort)
            if args.is_time_dependant:
                list_x[crossvalidation_set].append([])
                y[crossvalidation_set].append([])

            for j, psd_band_t_start_ms in enumerate(xrange(lower_bound, upper_bound, increment)):
                pack = _psd_time_unit_cut(args, i, j, raw, files_lim, total, psd_band_t_start_ms, upper_bound, shape, name)
                if pack is None:
                    continue
                shape, num_res_db = pack

                if args.is_time_dependant:
                    y[crossvalidation_set][-1].append(label)
                    list_x[crossvalidation_set][-1].append(num_res_db)

                else:
                    y[crossvalidation_set].append(label)
                    list_x[crossvalidation_set].append(num_res_db)

        assert len(list_x) == 3
        assert len(y) == 3

        # Make sure we have samples in each of the cross validation sets
        x_lens = [len(_x) for _x in list_x]
        for i, x_len in enumerate(x_lens):
            print((i, x_len))
            assert x_len > 0, "cross validation set #{} of x is empty. ".format(i)

        if not args.is_time_dependant:
            x = [None, None, None]
            for i in xrange(3):
                x[i] = np.dstack(list_x[i])
                print(x[i].shape)
                x[i] = x[i].astype(np.float32)       # We convert the PSD list of ndarrays to a single multidimensional ndarray
                x[i] = x[i].T  # Transpose for convenience

                y[i] = np.asarray(y[i], np.float32)  # We do the same with the labels

                no_samples_x = x[i].shape[utils.X_Dims.samples_and_times.value]
                # no_samples_y = y[i]

                # assert len(x[i].shape) == utils.X_Dims.size.value
                # assert no_samples_x == no_samples_y, (no_samples_x, no_samples_x)
                assert np.all(np.isfinite(x[i]))

        else:
            x = [[], [], []]

            assert len(list_x) > 0, len(list_x)
            for i in xrange(3):
                assert len(list_x[i]) > 0, len(list_x[i])


                for j in xrange(len(list_x[i])):
                    x[i].append(np.dstack(list_x[i][j]))
                    x[i][-1] = x[i][-1].astype(np.float32)
                    x[i][-1] = x[i][-1].T

                    y[i][j] = np.asarray(y[i][j], np.float32)

                    assert np.all(np.isfinite(x[i][-1]))
                    assert np.all(np.isfinite(y[i][-1]))

                no_samples_x = np.sum([x_f.shape[0] for x_f in x[i]])
                no_samples_y = np.sum([y_f.shape[0] for y_f in y[i]])

                assert no_samples_x == no_samples_y



        # Take any valid file's position information, as all raws [are supposed to] have the same positions.
        # Deep copying it allows the garbage collector to release the raw file. Not major at all.. but still.
        info = copy.deepcopy(next(data_gen(args.data_path))[1].info)
        print("--")
        print("Saving the newly generated dataset")
        saver_loader.save_ds((x, y, info))
        print("--")

    return x, y, info


def _aux_interp(sample_idx, x, sample_set_idx, no_samples, no_fft_bands, sensor_positions, grid, method, picks, show, interp_x):
    if sample_idx % 5 == 0:
        sys.stdout.write("\r\t- cv set: '{}', sample: {} of {} ({:4.2f} %))".format(
            sample_set_idx, sample_idx, no_samples, 100 * sample_idx / no_samples))
        sys.stdout.flush()

    buffer_shape = interp_x[sample_set_idx][sample_idx].shape
    buffer_ = np.empty(buffer_shape)
    for fft_channel_idx in xrange(no_fft_bands):
        point_values = x[sample_set_idx][sample_idx, fft_channel_idx, picks]
        buffer_[:, :, fft_channel_idx] = scipy.interpolate.griddata(sensor_positions, point_values, grid, method, 0)

    # It is said in h5py that h5py doesn't take numpy's fancy indexing very well. Even if most of it is supported,
    # it is still pretty slow.
    # As such, we have found that creating a numpy buffer and writing it in a continuous manner to the hdf5 file
    # is really much much faster.
    # This _really_ improves performance.
    interp_x[sample_set_idx][sample_idx] = buffer_


def make_interpolated_data(x, res, method, info, sensor_type, show, hdf5_saver_loader):
    # Take any valid file's position information, as all raws [are supposed to] have the same positions

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
    min_x = np.floor(np.min(sensor_positions[:, 0]))
    max_x = np.ceil(np.max(sensor_positions[:, 0]))
    min_y = np.floor(np.min(sensor_positions[:, 1]))
    max_y = np.ceil(np.max(sensor_positions[:, 1]))

    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, res[0],), np.linspace(min_y, max_y, res[1]))
    grid = (grid_x, grid_y)

    h5_f = h5py.File(hdf5_saver_loader.save_path, "w", libver="latest")
    interp_x = [None, None]
    acceptable = {"grad", "mag"}

    assert sensor_type in acceptable, \
        "sensor_type must be grad or mag, True (both) is not currently supported. Got {}.".format(sensor_type)

    parr = True # This is only put to False for debugging purposes

    with joblib.Parallel(n_jobs=32, backend="threading") as pool:
        for cv_set in range(2):
            no_samples = x[cv_set].shape[utils.X_Dims.samples_and_times.value]
            no_fft_bands = x[cv_set].shape[utils.X_Dims.fft_ch.value]

            shape = [no_samples, res[0], res[1], no_fft_bands]
            interp_x[cv_set] = h5_f.create_dataset(str(cv_set), shape, np.float32)

            constant_args = dict(x=x, sample_set_idx=cv_set, no_samples=no_samples, no_fft_bands=no_fft_bands,
                                 sensor_positions=sensor_positions, grid=grid, method=method, picks=picks,
                                 show=show, interp_x=interp_x)

            if parr:
                pool(joblib.delayed(_aux_interp)(sample_idx=sample_idx, **constant_args)
                     for sample_idx in xrange(no_samples))

            else:
                start = time.time()
                for sample_idx in xrange(no_samples):
                    _aux_interp(sample_idx=sample_idx, **constant_args)

                    if sample_idx % 1000 == 0 and sample_idx != 0:
                        sys.stderr.write("\ntook {} s for 1000\n".format(time.time() - start))
                        sys.stderr.flush()
                        start = time.time()

            assert np.all(np.isfinite(interp_x[cv_set]))

            print(interp_x[cv_set].shape)
    h5_f.close()
