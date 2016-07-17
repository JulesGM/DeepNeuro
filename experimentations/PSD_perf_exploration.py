from __future__ import with_statement, print_function

from six import iteritems
from six.moves import zip as izip

import sys, os, argparse, timeit, time, logging
from subprocess import check_output
from collections import defaultdict as dd, Counter

import numpy as np
import mne, mne.time_frequency
from mne.decoding import CSP

"""
mne's logger is way to talkative, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) work.
"""
logger = logging.getLogger('mne')
logger.disabled = True

# Add module root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import *
verbose = 0

def err_print(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def reg_print(msg, verbose=None):
    print(msg)

def timed_psd_log_info(infos_dict):
    """
    Wanted to remove the logging abstractions from the computing abstractions.
    """
    if verbose > 2:
        print("---------------------------------------------------------")
        for k, v in iteritems(infos_dict):
            print(("%s:" % k).ljust(10, " ") + "'%s'" % v)
        print("---------------------------------------------------------")

def timed_psd(f, rep, **f_fkwargs):
    """
    Time the psd function, log a couple things.
    At the moment of writing this comment, the primary candidate for this is mne.time_frequency.psd_welsh, because of its
    much better performance over mne.time_frequency.psd_multitaper
    f_fkwargs are meant to be f's kwargs.
    """

    # In Python, lists are the closest thing to a pointer we have access to. Here, we use a list as a ref
    # to collect the results of the timed function, which aren't returned by the timeit function (it returns the mesured
    # execution times).
    results = []

    # In my opinion this is cleaner than a lambda LOL WTF IS THIS
    def run():
        results.append(f(**f_fkwargs))

    chrono = np.array(timeit.Timer(run).repeat(rep, 1))

    loggable_info = {}

    loggable_info["avg"]        =  np.average(chrono)
    loggable_info["min"]        = np.min(chrono)
    loggable_info["f.__name__"] = f.__name__
    loggable_info["rep"]        = rep

    loggable_info.update(f_fkwargs)
    timed_psd_log_info(loggable_info)
    psd_amps, psd_freqs = results[0]

    return chrono, psd_amps

"""
http://martinos.org/mne/stable/generated/mne.time_frequency.psd_welch.html#mne.time_frequency.psd_welch

Current plan:
    - toCONFandHDF5:
        - Make a small lib that saves a conf with an hdf5 file,
            and regenerates & saves if any of the conf has changed
    - PSDs to HDF5 :
        - toCONFandHDF5
    - INTERPOLATION:    return linear_X, linear_Ylet let let     return linear_X, linear_Ylet let let


        SIZE preoccupations:
        - toCONFandHDF5
        - Similarly, if interp conf hasn't changed, load from hdf5
        - Else, load from hdf5
        ----> THIS WILL BE OF HUGE SIZE... res**2 * nfft
        - we'll see how slow to generate this is
    - DEEPLEARNING:
        ------>
"""

import enum
class X_Dims(enum.Enum):
    samples_and_times = 0
    fft_ch = 1
    sensors = 2
    size = 3


def maybe_prep_psds(argv):
    # Doing the arg parsing here is horrible. This is for this iteration only.
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################
    # DEFAULTS FOR THE ARGUMENTS
    d_REP        = 1  # Number of repetitions, for timeit
    d_GLOB_TINCR = 10 # Length of the time clips on which the PSDs are calculated
    d_NOVERLAP   = 0  # Overlap between the time clips on which the PSDs are calculated
    d_NFFT       = 256    # Quantity of fft bands

    # Defaults that are used
    ## welch is much faster, like 50x faster. There is more leakage than multitaper, but the performance hit
    ## is way too big for the moment for it to be justified.
    d_F = "welch"
    ## procs_to_use: 1 is the fastest for our data size for both our pc and the cluster... ##
    d_MIN_PROCS = 1
    d_MAX_PROCS = 1
    # we're not dumping any part of the features
    d_GLOB_TMIN = 0
    d_GLOB_TMAX = 1000


    # ARGUMENT PARSING
    p = argparse.ArgumentParser(argv)
    p.add_argument(       "--nfft",        type=int,  default=d_NFFT)
    p.add_argument(       "--glob_tincr",  type=int,  default=d_GLOB_TINCR)
    p.add_argument(       "--noverlap",    type=int,  default=d_NOVERLAP)
    p.add_argument("-o",  "--data_path",   type=str)

    # These don'psd_time_band_start need to be explored anymore (for the moment at least).
    p.add_argument("-r",  "--reps",         type=int, default=d_REP)
    p.add_argument(       "--min_procs",   type=int,  default=d_MIN_PROCS)
    p.add_argument(       "--max_procs",   type=int,  default=d_MAX_PROCS)
    p.add_argument("-f",  "--funcs",       type=str,  default=d_F)
    p.add_argument(       "--glob_tmin",   type=int,  default=d_GLOB_TMIN)
    p.add_argument(       "--glob_tmax",   type=int,  default=d_GLOB_TMAX)

    args = p.parse_args()

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
    REP          = args.reps
    FUNCS        = args.funcs
    NFFT         = args.nfft
    GLOB_TMIN    = args.glob_tmin
    GLOB_TMAX    = args.glob_tmax
    GLOB_TINCR   = args.glob_tincr
    NOVERLAP     = args.noverlap
    DATA_PATH    = args.data_path

    PSD_FUNC     = mne.time_frequency.psd_welch

    # We convert the values of FUNC to actual functions
    if FUNCS == "both":
        FUNCS_list = [mne.time_frequency.psd_multitaper, mne.time_frequency.psd_welch]
    elif FUNCS == "welch" or FUNCS == "w":
        FUNCS_list = [mne.time_frequency.psd_welch]
    elif FUNCS == "mt" or FUNCS == "multitaper" or FUNCS == "m":
        FUNCS_list = [mne.time_frequency.psd_multitaper]
    else:
        raise RuntimeError("invalid funcs argument. must be one of ['both', 'w', 'm']. Got '%s'." % FUNCS)

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################
    print("# FEATURE PREPARATION")

    # We want to time the compute time, not the data loading time.
    # The raw data is at preload = True to make this measure more relevant (less dependant on the HDD reads)
    start = time.time()

    checked_min_procs = MIN_PROCS if MIN_PROCS else 1 # 1 if MIN_PROCS is 0 or None

    X = None # Generated PSDs
    Y = [] # Generated labels

    for raw, label in data_gen(DATA_PATH):
            outer_time_bound = raw.n_times / 1000.
            for procs_to_use in range(checked_min_procs, MAX_PROCS + 1):
                for psd_band_t_start in range(GLOB_TMIN, GLOB_TMAX + 1, GLOB_TINCR):

                    if outer_time_bound < psd_band_t_start:
                        reg_print("{} < {} ; rejected".format(outer_time_bound, psd_band_t_start))
                        break

                    # So, the point here is that we don't want to crash if the raw is malformed.
                    # However, just catching all ValueError's is really too permissive; we need to be more precise here.
                    time_res, num_res = timed_psd(PSD_FUNC, REP, n_jobs=procs_to_use,
                                                  inst=raw, picks=mne.pick_types(raw.info, meg=True), n_fft=NFFT,
                                                  n_overlap=NOVERLAP, tmin=psd_band_t_start,
                                                  tmax=psd_band_t_start + GLOB_TINCR, verbose="INFO")

                    if X is None:
                        X = num_res
                    else:
                        if num_res.shape[X_Dims.fft_ch.value] == X.shape[X_Dims.fft_ch.value]: # All samples need the same qty of fft channels
                            X = np.dstack([X, num_res])
                        else:
                            print("num_res of bad shape '{}' rejected, should be {}".format(num_res.shape, X.shape[:2]))
                            continue

                    Y.append(label)


    assert X is not None, "X is None"

    # We convert the PSD list of ndarrays to a single multidimensional ndarray
    X = X.astype(np.float32)
    assert type(X) == np.ndarray and X.dtype == np.float32
    # We do the same with the labels
    Y = np.asarray(Y, np.float32)
    X = X.T

    assert len(X.shape) == X_Dims.size.value
    assert X.shape[X_Dims.samples_and_times.value] == Y.shape[0], X.shape[X_Dims.samples_and_times.value]  # no_samples
    # assert X.shape[1] == 129, X.shape[1] # ~ nfft: 129 lectures for each of
    assert X.shape[X_Dims.sensors.value] == 306, X.shape[X_Dims.sensors.value]  # sensor no
    #assert X.shape[X_Dims.fft_ch] = ? #

    print("X.shape = {}".format(X.shape))
    print("Y.shape = {}".format(Y.shape))

    return X, Y


def make_picks(no_samples):
    ###########################################################################
    # DATA SPLIT
    ###########################################################################
    print("# DATA SPLIT")

    training_limit = int(.6 * no_samples)
    valid_limit = training_limit + int(.2 * no_samples)
    randomized = np.random.permutation(no_samples)
    training_picks = randomized[:training_limit]
    valid_picks = randomized[training_limit:valid_limit]
    test_picks = randomized[valid_limit:]
    return training_picks, valid_picks, test_picks


def make_samples_linear(X, Y):
    linear_X = X.reshape(X.shape[X_Dims.samples_and_times.value],  X.shape[X_Dims.fft_ch.value] * X.shape[X_Dims.sensors.value])
    return linear_X, Y


def linear_classification(linear_X, linear_Y, train_picks, valid_picks, test_picks):
    ## Temporary, clean split in the error output

    header = ("*********************************************************\n"
              "**** Classification Code :                               \n"
              "*********************************************************")
    err_print(header)
    reg_print(header)

    assert len(linear_X.shape) == 2
    from sklearn.svm import SVC
    from sklearn.linear_model import logistic
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
    from sklearn.ensemble import RandomForestClassifier


    # We currently only use sklearn classifiers and they all use the same interface, so we just build
    # the classifiers in a list and try them iteratively.
    # Eventually, we will have more precise configs for them, undoubtably
    classifiers = [
        SVC(kernel="linear"),
        logistic.LogisticRegression(),
        RandomForestClassifier(),
        # lda()
    ]

    for classifier in classifiers:
        reg_print(linear_X.dtype)
        reg_print(linear_X.shape)
        reg_print(linear_Y.dtype)
        reg_print(linear_Y.shape)
        classifier.fit(linear_X[train_picks, :], linear_Y[train_picks])

    trainning_pred = {}
    trainning_succ = {}

    for classifier in classifiers:
        trainning_pred[classifier] = classifier.predict(linear_X[valid_picks, :])
        trainning_succ[classifier] = np.average(trainning_pred[classifier] == linear_Y[valid_picks])

        reg_print("classifier:\n\t- {classifier}\n\t- {success}\n\t- {score}"
            .format(
                classifier=classifier,
                success=trainning_succ[classifier],
                score=classifier.score(linear_X[valid_picks, :], linear_Y[valid_picks])))

def main(argv):
    start = time.time()

    X, Y = maybe_prep_psds(argv) # argv being passed is temporary

    ###########################################################################
    # CLASSICAL MACHINE LEARNING CLASSIFICATION without locality
    ###########################################################################

    reg_print("# CLASSICAL MACHINE LEARNING")

    linear_X, linear_Y = make_samples_linear(X, Y)
    linear_training_picks, linear_valid_picks, linear_test_picks = make_picks(linear_X.shape[0])
    linear_classification(linear_X, linear_Y, linear_training_picks, linear_valid_picks, linear_test_picks)

    ###########################################################################
    # LOCALITY PRESERVING CLASSICAL MACHINE LEARNING
    ###########################################################################

    ###########################################################################
    # VGG classical style CONVOLUTIONAL NEURAL NETWORK
    ###########################################################################

    # 3x3 conv, relu, 3x3 conv, relu, 3x3 conv, relu, maxpool, 3x3 conv, relu, 3x3 conv, relu, maxpool, FC, FC
    # with batchnorm and dropout

    # TODO

    ###########################################################################
    # RESNET CONVOLUTIONAL NEURAL NETWORK
    ###########################################################################

    # TODO
    end = time.time()
    reg_print("TOTAL TIME: {total_time} sec".format(total_time=end - start))
    reg_print("*********************************************************\n"
          "Total: '%s'" % (end - start))

if __name__ == "__main__": main(sys.argv)
