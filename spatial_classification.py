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

base_path = os.path.join(os.path.dirname(__file__))
def spatial_classification(x, y, res, nfft, tincr, use_established_bands, info, job_type):
    descript = {}
    descript["res"] = res
    descript["nfft"] = nfft
    descript["use_established_bands"] = use_established_bands
    descript["tincr"] = tincr

    # make a hash of the description for the filename, that way we know to load a save if it has the same hash as a name,
    # or to compute the interpolations if not
    name = tuple(sorted(descript.items(), key=lambda x: x[0])).__hash__()
    saver_loader = utils.data_utils.SaverLoader(os.path.join(base_path, "saves/interp_image_saves/{}.pkl".format(name)))

    if saver_loader.save_exists():
        prepared_x, descript = saver_loader.load_ds()
    else:
        prepared_x, descript = NN_models.make_interpolated_data(x, res, "cubic", info, )
        saver_loader.save_ds((prepared_x, descript))

    for i in xrange(2):
        y[i] = utils.to_one_hot(y[i], np.max(y[i]) + 1)

    training_prepared_x   = prepared_x[0]
    training_y   = y[0]

    validation_prepared_x = prepared_x[1]
    validation_y = y[1]

    # print(training_prepared_x)
    x_shape   = training_prepared_x.shape
    y_shape_1 = training_y.shape[1]


    print(x_shape)
    print(y_shape_1)

    model = NN_models.CNN(x_shape, y_shape_1, depth=3, dropout_keep_prob=.9, filter_scale_factor=1.5)
    model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                            n_epochs=10000000,
                            minibatch_size=128, learning_rate=0.001)