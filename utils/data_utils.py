from __future__ import print_function, generators, division, with_statement
from six import iteritems
import os, sys, re, glob, warnings, logging, enum, json
import mne.io.pick
import mne
import numpy as np
from utils import *

mne.set_log_level("ERROR")

def data_gen(base_path, limit = None):
    """

    The objective is to never have exceptions, to always know what to ignore and why.

    Test script:
        to helios 1>/dev/null; ssh helios 'cd COCO; python -c "from data_utils import data_gen; [x for x in data_gen()]"'

    """

    base_path = os.path.abspath(base_path)
    assert os.path.exists(base_path), "{base_path} doesn't exist".format(base_path=base_path)
    full_glob = glob.glob(base_path + "/*.fif")

    if len(full_glob) == 0:
        raise RuntimeError("Datagen didn't find find any '.fif' files")

    print("glob found {} .fif files".format(len(full_glob)))

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
        name = fif_path.split("/")[-1] # os.path.split appears broken somehow

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

        assert name.lower().startswith("k") or name.lower().startswith("r"), "file name is weird, can't guess label from it. ({})".format(name)
        label = name.lower().startswith("k")

        yield name, raw, label


def maybe_prep_psds(args):

    # Display the args
    print("\nArgs:" )
    for k, v in iteritems(vars(args)):
        print("--{k}:".format(k=k).ljust(20, " ") + "{v}".format(v=v))
    print("")

    # Warn if some values are weird
    if args.min_procs != 1 or args.max_procs != 1:
        print("Warning: --min_procs and --max_procs should probably both be 1, " \
              "or left alone, as benchmarks say more procs decrease performance.")

    if args.reps != 1:
        print("Warning: --rep should be 1 or left alone, unless you want to test the " \
              "performance of the psd function, which there is no real reason to do right now."\
              "Value: {}".format(args.reps))

    if args.glob_tmin != 0:
        print("Warning: --glob_tmin is not equal to zero, this is weid. Value : {}".format(args.glob_tmin))

    # We assign the values we obtained
    MIN_PROCS    = args.min_procs
    MAX_PROCS    = args.max_procs
    NFFT         = args.nfft
    GLOB_TMIN_s    = args.glob_tmin
    GLOB_TMAX_s    = args.glob_tmax
    GLOB_TINCR_s   = args.glob_tincr
    NOVERLAP     = args.noverlap
    DATA_PATH    = args.data_path

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################
    print("# FEATURE PREPARATION")

    checked_min_procs = MIN_PROCS if MIN_PROCS else 1 # 1 if MIN_PROCS is 0 or None

    X = [None, None, None] # Generated PSDs
    Y = [[], [], []] # Generated labels

    BASE_PATH = os.path.dirname(os.path.dirname(__file__))
    json_path = os.path.join(BASE_PATH, "fif_split.json")
    print("#2: {}".format(json_path))
    with open(json_path, "r") as json_f:
        fif_split = json.load(json_f) # dict with colors

    split_idx_to_name = {"training":  0,
                         "valid":     1,
                         "test":      2,
                         }


    procs_to_use = 1
    GLOB_TINCR_s = 0.1

    freqs_bands = None

    for name, raw, label in data_gen(DATA_PATH):
        split_idx = split_idx_to_name[fif_split[name]]


        ended_at_ms = None # Python 3 scope rules

        lower_bound = GLOB_TMIN_s * 1000
        increment = int(GLOB_TINCR_s * 1000)
        # we ignore GLOB_TMAX_s if it's later than the end of the measure
        delta = min(raw.n_times, GLOB_TMAX_s * 1000) - lower_bound
        # The upper bound is the largest number of samples that allows for complete non overlapping PSD evaluation windows
        # It's the total number of samples minus the rest of the division of the total number of samples by the PSD
        # evaluation window width (the mod).
        # If building something takes exactly 10 minutes, you have 45 minutes but only want full things to be built,
        # then you will be done in
        # 45 - 45 % 10 = 45 - 5 = 40 minutes
        # if you start at 0h10,
        # you will be done at
        # 0h10 + 45 - 45 % 10 = 0h50 (this last part is pretty obvious)
        upper_bound = lower_bound + delta - delta % int(GLOB_TINCR_s * 1000)
        # previous upper bound : min(1000 * GLOB_TMAX_s, raw.n_times)
        for psd_band_t_start_ms in range(lower_bound, upper_bound, increment):
            """
            So, GLOB_* are in seconds, and raw.n_times is in milliseconds.
            Iterating on a range requires ints, and we don't want to lose precision by rounding up the milliseconds to seconds,
            so we put everything in milliseconds.

            mne.time_frequency.psd_welch takes times in seconds, but accepts floats, so we divide by 1000 while
            keeping our precision.
            """

            num_res_db, freqs = mne.time_frequency.psd_welch(
                                       n_jobs=1, # in our tests, more jobs invariably resulted in slower execution, even on the 32 cores xeons of the Helios cluster.
                                       inst=raw,
                                       picks=mne.pick_types(raw.info, meg=True),
                                       n_fft=NFFT,
                                       n_overlap=NOVERLAP,
                                       tmin=psd_band_t_start_ms / 1000.,
                                       tmax=psd_band_t_start_ms / 1000. + GLOB_TINCR_s,
                                       verbose="INFO"
                                       )
            if freqs_bands is None:
                freqs_bands = freqs
                print(freqs)
            else:
                assert np.all(freqs_bands == freqs), "Frequency bands are supposed to be consistant across all samples.\n\nPreviously saved:\n\t{}\n\ngot:\n\t{}\n\n".format(freqs_bands, freqs)

            num_res_db = 10.0 * np.log10(num_res_db)

            if X[split_idx] is None:
                X[split_idx] = num_res_db
            else:
                assert num_res_db.shape[X_Dims.fft_ch.value] == X[split_idx].shape[X_Dims.fft_ch.value], "A sample has a weird shape. This shouldn't happen (anymore). This used to mean that we were trying to evaluate the PSD on a smaller window that the one we used on the previous units of the batch."
                X[split_idx] = np.dstack([X[split_idx], num_res_db])

            Y[split_idx].append(label)


    assert len(X) == 3
    assert len(Y) == 3

    for i in xrange(3):
        assert type(X[i]) == np.ndarray

        # We convert the PSD list of ndarrays to a single multidimensional ndarray
        X[i] = X[i].astype(np.float32)

        # We do the same with the labels
        Y[i] = np.asarray(Y[i], np.float32)

        # Transpose for convenience
        X[i] = X[i].T

        # Center and normalise
        X[i] = (X[i] - np.mean(X[i]))
        X[i] = X[i] / np.std(X[i])

        assert len(X[i].shape) == X_Dims.size.value
        assert X[i].shape[X_Dims.samples_and_times.value] == Y[i].shape[0], X[i].shape[X_Dims.samples_and_times.value]  # no_samples
        assert X[i].shape[X_Dims.sensors.value] == 306, X[i].shape[X_Dims.sensors.value]  # sensor no

        print("X[{}].shape = {}".format(i, X[i].shape))
        print("Y[{}].shape = {}".format(i, Y[i].shape))

    # Take any valid file's position information, as all raws [are supposed to] have the same positions
    info = next(data_gen(DATA_PATH))[1].info

    return X, Y, info