#! /usr/bin/env python
# Compatibility imports
from __future__ import with_statement, print_function, division
from six import iteritems
from six.moves import zip as izip

# stdlib usual imports
import sys, os, argparse, time, logging, enum, json, subprocess as sp
from collections import defaultdict as dd, Counter

import utils.data_utils
import utils

import linear_classification as LC
import spatial_classification as SC

import numpy as np
import tensorflow as tf


"""
MNE's logger prints massive amount of useless stuff, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) seem to be working.
"""
logger = logging.getLogger('mne')
logger.disabled = True


def parse_args(argv):
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################

    # ARGUMENT PARSING
    p = argparse.ArgumentParser(argv)
    p.add_argument("--nfft",            type=int,)
    p.add_argument("--glob_tincr",      type=int,)
    p.add_argument("--limit",           type=int, default=None)
    p.add_argument("-o", "--data_path", type=str,)

    p.add_argument("--job_type",        type=str, default="NN")
    p.add_argument("--noverlap",        type=int, default=0)
    p.add_argument("--glob_tmin",       type=int, default=0)
    p.add_argument("--glob_tmax",       type=int, default=1000000)

    return p.parse_args(argv[1:])


def main(argv):

    BASE_PATH = os.path.dirname(__file__)
    argv = [None,
            "--nfft",         "4",
            "--glob_tincr",   "10",
            "--data_path",    "/home/julesgm/aut_gamma/",
            ]

    args = parse_args(argv)

    print("\nArgs:")
    for key, value in iteritems(vars(args)):
        print("\t- {:12}: {}".format(key, value))
    print("--")
    json_split_path = "./fif_split.json"

    if not os.path.exists(json_split_path):
        import generate_split
        generate_split.main([None, args.data_path, BASE_PATH])

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

    import sklearn.preprocessing
    scaler = sklearn.preprocessing.StandardScaler()
    linear_x[0] = scaler.fit_transform(linear_x[0])
    linear_x[1] = scaler.transform(linear_x[1])
    linear_x[2] = scaler.transform(linear_x[2])

    LC.linear_classification(linear_x, Y, args.job_type)

    return 0

    res_x = 10
    res_y = 10
    interp = "cubic"
    saver_loader = utils.data_utils.SaverLoader("./interp_image_saves/{interp}_{resx}_{resy}_{limit}_{glob_tincr}_{nfft}_{name}_images.pkl" \
                        .format(interp=interp, res_x=res_x, res_y=res_y, limit=args.limit,
                                glob_tincr=args.glob_tincr, nfft=args.nfft, name="images"))

    if saver_loader.save_exists():
        interp_x = saver_loader.load_ds()
    else:
        interp_x = [None, None, None]
        for i in range(3):
            interp_x[i] = SC.make_interpolated_data(X[i], (res_x, res_y), interp, sample_info)

        import custom_cells.tensorflow_resnet as tf_resnet
        import custom_cells.tensorflow_resnet.resnet_train as tf_resnet_train

        is_training = tf.placeholder('bool', [], name='is_training')

        images, labels = tf.cond(is_training,
                                 lambda: (interp_x[0], Y[0]),
                                 lambda: (interp_x[1], Y[1]))

        saver_loader.save_ds(interp_x)

    logits = tf_resnet.inference_small(images, num_classes=2, is_training=is_training, use_bias=False, num_blocks=2)
    tf_resnet_train.train(is_training, logits, images, labels)

    #spatial_classification(interp_X, Y, training_picks, valid_picks, test_picks)

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
