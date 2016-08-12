# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

import utils
import utils.data_utils
import NN_models

import numpy as np


def prepare_data(x, info):
    prepared_x, descript = NN_models.make_interpolated_data(x, (10, 10), info, )
    return prepared_x, descript

def spatial_classification(x, y, res, nfft, tincr, use_established_bands, info, job_type):
    descript = {}
    descript["res"] = res
    descript["nfft"] = nfft
    descript["use_established_bands"] = use_established_bands
    descript["tincr"] = tincr

    # make a hash of the description for the filename, that way we know to load a save if it has the same hash as a name,
    # or to compute the interpolations if not
    name = tuple(sorted(descript.items(), key=lambda x: x[0])).__hash__()

    saver_loader = utils.data_utils.SaverLoader("/home/julesgm/COCO/saves/interp_image_saves/{}.pkl".format(name))

    if saver_loader.save_exists():
        prepared_x, descript = saver_loader.load_ds()
    else:
        prepared_x = prepare_data(x, info)
        saver_loader.save_ds((prepared_x, descript))

    no_samples = x.shape[0]
    indices = np.random.permutation(np.arange(no_samples))
    lim_tr = int(.65 * no_samples)
    lim_va = lim_tr + int(.18 * no_samples)

    training_idx = indices[:lim_tr]
    validation_idx = indices[lim_tr:lim_va]
    test_idx = indices[lim_va:]

    training_prepared_x = prepared_x[training_idx]
    training_y = y[training_idx]
    validation_prepared_x = prepared_x[validation_idx]
    validation_y = y[validation_idx]

    model = NN_models.SmallResNet(training_prepared_x, training_y, validation_prepared_x, validation_y, 1,
                                  0.001, 512)
    model.fit()