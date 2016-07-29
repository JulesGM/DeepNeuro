#! /usr/bin/env python

# Compatibility imports
from __future__ import with_statement, print_function, division
from six import iteritems
from six.moves import zip as izip

# Stdlib imports
import sys, os, argparse, time, logging, enum, json, subprocess as sp
from collections import defaultdict as dd, Counter

from utils.data_utils import *
from utils import *
from linear_classification import *
from spatial_classification import *

# scipy/numpy/matplotlib/tf
import numpy as np, scipy
import h5py

"""
mne's logger is way to talkative, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) work.
"""
logger = logging.getLogger('mne')
logger.disabled = True


def parse_args(argv):
    # Doing the arg parsing here is horrible. This is for this iteration only.
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################
    # DEFAULTS FOR THE ARGUMENTS
    d_REP = 1  # Number of repetitions, for timeit
    d_GLOB_TINCR = 10  # Length of the time clips on which the PSDs are calculated
    d_NOVERLAP = 0  # Overlap between the time clips on which the PSDs are calculated
    d_NFFT = 2048  # Quantity of fft bands

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

    d_JOB_TYPE = "NN"

    # ARGUMENT PARSING
    p = argparse.ArgumentParser(argv)
    p.add_argument("--nfft", type=int, default=d_NFFT)
    p.add_argument("--glob_tincr", type=int, default=d_GLOB_TINCR)
    p.add_argument("--noverlap", type=int, default=d_NOVERLAP)
    p.add_argument("-o", "--data_path", type=str)

    # These don'psd_time_band_start need to be explored anymore (for the moment at least).
    p.add_argument("-r", "--reps", type=int, default=d_REP)
    p.add_argument("--min_procs", type=int, default=d_MIN_PROCS)
    p.add_argument("--max_procs", type=int, default=d_MAX_PROCS)
    p.add_argument("-f", "--funcs", type=str, default=d_F)
    p.add_argument("--glob_tmin", type=int, default=d_GLOB_TMIN)
    p.add_argument("--glob_tmax", type=int, default=d_GLOB_TMAX)

    p.add_argument("--job_type", type=str, default=d_JOB_TYPE)

    return p.parse_args(argv[1:])


def main(argv):
    start = time.time()
    BASE_PATH = os.path.dirname(__file__)

    # If we don't have arguments, try to load last known config.
    # If we do have argumetns, save them as the last known config.
    if len(argv) <= 1:
        with open(os.path.join(BASE_PATH, "direct_args.json"), "r") as _if:
            argv = json.load(_if)
    else:
        with open(os.path.join(BASE_PATH, "direct_args.json"), "w") as _if:
            json.dump(argv, _if)

    args = parse_args(argv)


    json_split_path = os.path.join(BASE_PATH, "fif_split.json")
    print("#1: {}".format(json_split_path))

    if not os.path.exists(json_split_path):
        import generate_split
        generate_split.main([None, args.data_path, BASE_PATH])

    X, Y, sample_info = maybe_prep_psds(args) # argv being passed is temporary

    for i in xrange(3):
        print("")
        print("overall X stddev: {}".format(np.average(X[i])))
        print("overall X stddev: {}".format(np.std(X[i])))
        print("")

    ###########################################################################
    # CLASSICAL MACHINE LEARNING CLASSIFICATION without locality
    ###########################################################################

    print("################################################################")
    print("# CLASSICAL MACHINE LEARNING")
    print("################################################################")
    linear_x = [None, None, None]
    linear_Y = [None, None, None]

    print("# Information about the linearized dataset")
    names=["training", "validation", "testing"]
    for i in xrange(3):
        linear_x[i], linear_Y[i] = make_samples_linear(X[i], Y[i])
        print("#\tshape of linear_x {}: {}".format(names[i], linear_x[i].shape))
        print("#\tstddev: {}\n\tmean: {}".format(np.std(linear_x[i]), np.mean(linear_x[i])))

    linear_classification(linear_x, linear_Y, args.job_type)

    # reg_print("# SPATIAL MACHINE LEARNING")
    # interp_X = make_interpolated_data(X, (1000, 1000), "cubic", sample_info)

    # spatial_classification(interp_X, Y, training_picks, valid_picks, test_picks)

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
    print("TOTAL TIME: {total_time} sec".format(total_time=end - start))
    print("*********************************************************\n"
          "Total: '%s'" % (end - start))

if __name__ == "__main__": main(sys.argv)
