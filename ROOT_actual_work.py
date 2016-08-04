#! /usr/bin/env python
# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

# stdlib usual imports
import sys
import os
import argparse
import time
import logging
import warnings
import utils
import utils.data_utils

import linear_classification as LC

import sklearn.preprocessing
import numpy as np

"""
MNE's logger prints massive amount of useless stuff, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) seem to be working.
"""
logger = logging.getLogger('mne')
logger.disabled = True


def parse_args(argv):
    p = argparse.ArgumentParser(argv)
    p.add_argument("--nfft",            type=int, default="5")
    p.add_argument("--glob_tincr",      type=float, default="0.1")
    p.add_argument("--job_type",        type=str, default="NN")

    p.add_argument("--limit",           type=int, default=None)
    p.add_argument("-o", "--data_path", type=str, default=os.path.join(os.environ["HOME"], "aut_gamma"))
    p.add_argument("--noverlap",        type=int, default=0)
    p.add_argument("--glob_tmin",       type=int, default=0)
    p.add_argument("--glob_tmax",       type=int, default=1000000)

    return p.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)

    if six.PY3:
        print("The code hasn't been tested in Python 3.\n")

    print("\nArgs:")
    for key, value in six.iteritems(vars(args)):
        print("\t- {:12}: {}".format(key, value))
    print("--")

    json_split_path = "./fif_split.json"

    if not os.path.exists(json_split_path):
        raise RuntimeError("Couldn't find fif_split.json. Should be generated with ./generate_split.py at the beginning"
                           " of the data exploration, and then shared.")

    X, Y, sample_info = utils.data_utils.maybe_prep_psds(args)

    print("Dataset properties:")
    for i in xrange(3):
        print("\t- {} nan/inf:    {}".format(i, np.any(np.isnan(X[i]))))
        print("\t- {} shape:      {}".format(i, X[i].shape))
        print("\t- {} mean:       {}".format(i, np.mean(X[i])))
        print("\t- {} stddev:     {}".format(i, np.std(X[i])))
        print("\t--")
    print("--")

    ###########################################################################
    # CLASSICAL MACHINE LEARNING CLASSIFICATION without locality
    ###########################################################################
    linear_x = [None, None, None]

    for i in xrange(3):
        linear_x[i] = LC.make_samples_linear(X[i])

    scaler = sklearn.preprocessing.StandardScaler()
    linear_x[0] = scaler.fit_transform(linear_x[0])
    linear_x[1] = scaler.transform(linear_x[1])
    linear_x[2] = scaler.transform(linear_x[2])

    LC.linear_classification(linear_x, Y, args.job_type)

    ###########################################################################
    # LOCALITY PRESERVING CLASSICAL MACHINE LEARNING
    ###########################################################################

    ###########################################################################
    # VGG classical style CONVOLUTIONAL NEURAL NETWORK
    ###########################################################################

    # 3x3 conv, relu, 3x3 conv, relu, 3x3 conv, relu, maxpool, 3x3 conv, relu, 3x3 conv, relu, maxpool, FC, FC
    # with batchnorm and dropout

    ###########################################################################
    # RESNET CONVOLUTIONAL NEURAL NETWORK
    ###########################################################################

    # spatial_classification(interp_X, Y, training_picks, valid_picks, test_picks)


if __name__ == "__main__": main(sys.argv)
