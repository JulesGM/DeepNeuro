from __future__ import print_function, generators, division, with_statement
from six import iteritems
import os, sys, re, glob, warnings, logging, enum, json
import mne.io.pick
import mne
import numpy as np
from utils import *


mne.set_log_level("ERROR")

import sklearn.preprocessing
import joblib


class SaverLoader(object):
    def __init__(self, path):
        self._save_path = path

    def save_ds(self, data):
        joblib.dump(data, self._save_path)

    def load_ds(self):
        return joblib.load(self._save_path)

    def save_exists(self):
        return os.path.exists(self._save_path)


def data_gen(base_path, limit=None):
    """
    Generator
    Yields raw files
        -- name, raw, label, len(full_glob)
    """

    base_path = os.path.abspath(base_path)
    assert os.path.exists(base_path), "{base_path} doesn't exist".format(base_path=base_path)
    full_glob = glob.glob(base_path + "/*.fif")

    if len(full_glob) == 0:
        raise RuntimeError("Datagen didn't find find any '.fif' files")

    if limit is not None:
        print(">>>>>>>>>> Warning: data_gen limit argument is not None.\n"
              ">>>>>>>>>> This has the effect that only a limited amount ({})\n"
              ">>>>>>>>>> of the data will be loaded. \n".format(limit))

    fif_paths = full_glob[:limit] if limit is not None else full_glob

    if len(fif_paths) == 0:
        raise RuntimeError("fif_path is of size zero.")

    failed = 0
    for fif_path in fif_paths:
        logging.info("Ignored ratio: {}" .format(failed / len(fif_paths)))
        name = fif_path.split("/")[-1]

        # mne generates a lot of needless blabla
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = mne.io.Raw(fif_path)
            except ValueError, err:
                logging.error("-- data_gen ValueError:")
                logging.error("-- %s" % name)
                logging.error("-- %s\n" % err)
                raise err
            except TypeError, err:
                logging.error("-- data_gen TypeError")
                logging.error("-- %s" % name)
                logging.error("-- %s\n" % err)
                raise err

        assert name.lower().startswith("k") or name.lower().startswith("r"), \
            "file name is weird, can't guess label from it. ({})".format(name)

        label = name.lower().startswith("k")

        yield name, raw, label, len(full_glob)


def maybe_prep_psds(args):
    limit = args.limit

    if args.glob_tmin != 0:
        print("Warning: --glob_tmin is not equal to zero, this is weid. Value : {}".format(args.glob_tmin))

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################

    X = [None, None, None] # Generated PSDs
    Y = [[], [], []] # Generated labels

    base_path = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(base_path, "fif_split.json")

    with open(json_path, "r") as json_f:
        fif_split = json.load(json_f) # dict with colors

    split_idx_to_name = {"training":  0,
                         "valid":     1,
                         "test":      2,
                         }

    X = [None, None, None]

    # We build savepaths from different values of the parameters
    saver_loader = SaverLoader("/home/julesgm/COCO/ds_transform_saves/{limit}_{tincr}_{nfft}_latest_save.pkl" \
                                    .format(limit=limit, tincr=args.glob_tincr, nfft=args.nfft))

    if saver_loader.save_exists():
        print("Loading pickled dataset")
        X, Y, info = saver_loader.load_ds()
        print("--")

    else:
        print("Generating the dataset from the raw (.fif) files")
        freqs_bands = None
        list_x = [[], [], []]

        for i, (name, raw, label, total) in enumerate(data_gen(args.data_path, limit)):
            files_lim = total if limit is None or total > limit else limit

            split_idx = split_idx_to_name[fif_split[name]]

            lower_bound = args.glob_tmin * 1000
            increment = int(args.glob_tincr * 1000)
            # we ignore GLOB_TMAX_s if it's later than the end of the measure
            delta = min(raw.n_times, args.glob_tmax * 1000) - lower_bound
            # The upper bound is the largest number of samples that allows for complete non overlapping PSD evaluation windows
            # It's the total number of samples minus the rest of the division of the total number of samples by the PSD
            # evaluation window width (the mod).
            # If building something takes exactly 10 minutes, you have 45 minutes but only want full things to be built,
            # then you will be done in
            # 45 - 45 % 10 = 45 - 5 = 40 minutes
            # if you start at 0h10,
            # you will be done at
            # 0h10 + 45 - 45 % 10 = 0h50 (this last part is pretty obvious)
            upper_bound = lower_bound + delta - delta % int(args.glob_tincr * 1000)
            # previous upper bound : min(1000 * GLOB_TMAX_s, raw.n_times)

            for j, psd_band_t_start_ms in enumerate(range(lower_bound, upper_bound, increment)):
                """
                So, GLOB_* are in seconds, and raw.n_times is in milliseconds.
                Iterating on a range requires ints, and we don't want to lose precision by rounding up the milliseconds to seconds,
                so we put everything in milliseconds.

                mne.time_frequency.psd_welch takes times in seconds, but accepts floats, so we divide by 1000 while
                keeping our precision.
                """
                if j % 10 == 0:
                    sys.stdout.write("\r\t- File {} of {} (max: {}) - Segment {:7} of {:7}, {:<4.2f}%".format(
                        i + 1, files_lim, total, str(psd_band_t_start_ms), str(upper_bound),
                        100 * psd_band_t_start_ms / upper_bound))

                # in our tests, more jobs invariably resulted in slower
                # execution, even on the 32 cores xeons of the Helios cluster.
                num_res_db, freqs = mne.time_frequency.psd_welch(
                                           n_jobs=1,
                                           inst=raw,
                                           picks=mne.pick_types(raw.info, meg=True),
                                           n_fft=args.nfft,
                                           n_overlap=args.noverlap,
                                           tmin=psd_band_t_start_ms / 1000.,
                                           tmax=psd_band_t_start_ms / 1000. + args.glob_tincr,
                                           verbose="INFO"
                                           )

                num_res_db = 10.0 * np.log10(num_res_db)

                if not np.all(np.isfinite(num_res_db)):
                    print("\n>>>>>>>>> {} : has a NAN or INF or NINF post log - skipping this segment ({}:{})\n" \
                          .format(name, psd_band_t_start_ms, upper_bound))

                    continue

                list_x[split_idx].append(num_res_db)
                Y[split_idx].append(label)

                if freqs_bands is None:
                    freqs_bands = freqs

        print("")
        assert len(Y) == 3

        for i in xrange(3):
            X[i] = np.dstack(list_x[i])
            # We convert the PSD list of ndarrays to a single multidimensional ndarray
            X[i] = X[i].astype(np.float32)
            # We do the same with the labels
            Y[i] = np.asarray(Y[i], np.float32)
            # Transpose for convenience
            X[i] = X[i].T

            assert len(X[i].shape) == X_Dims.size.value
            assert X[i].shape[X_Dims.samples_and_times.value] == Y[i].shape[0], X[i].shape[X_Dims.samples_and_times.value]  # no_samples
            assert X[i].shape[X_Dims.sensors.value] == 306, X[i].shape[X_Dims.sensors.value]  # sensor no

        # Verify that all values are good
        for i in range(3):
            assert np.all(np.isfinite(X[i]))

        # Take any valid file's position information, as all raws [are supposed to] have the same positions
        info = next(data_gen(args.data_path))[1].info
        print("--")
        print("Saving the newly generated dataset")
        saver_loader.save_ds((X, Y, info))
        print("--")

    for x in range(3):
        print(Y[x][:])

    return X, Y, info