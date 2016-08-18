# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

import utils
import utils.data_utils
import NN_models

import numpy as np
import os

base_path = os.path.dirname(__file__)


def make_image_save_name(res, sensor_type, nfft, fmax, tincr, use_established_bands):
    # we right them all, on purpose, instead of using *args, to make sure everything is in its place
    args = [res, sensor_type, nfft, fmax, tincr, use_established_bands]
    return "_".join([str(x) for x in args]) + ".pkl"


def spatial_classification(x, y,  nfft, tincr, fmax, info, established_bands,  res, sensor_type,  net_type,
                           learning_rate, minibatch_size, dropout_keep_prob, depth, filter_scale_factor):

    saves_loc = os.path.join(base_path, "saves/interp_image_saves")
    if not os.path.exists(saves_loc):
        os.mkdir(saves_loc)

    image_save_name = make_image_save_name(res, sensor_type, nfft, tincr, fmax, established_bands)
    saver_loader = utils.data_utils.SaverLoader(os.path.join(saves_loc, image_save_name))

    if saver_loader.save_exists():
        prepared_x = saver_loader.load_ds()
    else:
        print("Preparing the images")
        prepared_x = NN_models.make_interpolated_data(x, res, "cubic", info)
        saver_loader.save_ds(prepared_x)

    for i in xrange(2):
        y[i] = utils.to_one_hot(y[i], np.max(y[i]) + 1)

    training_prepared_x = prepared_x[0]
    training_y = y[0]

    validation_prepared_x = prepared_x[1]
    validation_y = y[1]

    # print(training_prepared_x)
    x_shape = training_prepared_x.shape
    y_shape_1 = training_y.shape[1]

    print(x_shape)
    print(y_shape_1)

    if net_type == "cnn":
        print("cnn")
        model = NN_models.CNN(x_shape, y_shape_1, depth=depth, dropout_keep_prob=dropout_keep_prob,
                              filter_scale_factor=filter_scale_factor)
        model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y, n_epochs=10000000,
                              minibatch_size=minibatch_size, learning_rate=learning_rate)

    elif net_type == "resnet":
        print("resnet")
        model = NN_models.ResNet(x_shape, y_shape_1, depth=depth, dropout_keep_prob=dropout_keep_prob,
                                 filter_scale_factor=filter_scale_factor)
        model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                  n_epochs=10000000, minibatch_size=minibatch_size, learning_rate=learning_rate)
    else:
        raise RuntimeError("Received unexpected value '{}' for option --net_type".format(net_type))