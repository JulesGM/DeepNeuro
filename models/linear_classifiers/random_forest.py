# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

import utils
import numpy as np

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing

import subprocess
import concurrent.futures as futures
import ctypes
import numpy as np

from utils import make_samples_linear


def shared_array(copy_from):
    shared_array_base = multiprocessing.Array(ctypes.c_double, np.product(copy_from.shape))
    shared_array = np.frombuffer(shared_array_base.get_obj()).reshape(*copy_from.shape)
    shared_array = shared_array.reshape(copy_from.shape)
    shared_array[...] = copy_from[...]
    assert shared_array.base.base is shared_array_base.get_obj()

    return shared_array


def chain(cl, training_x, training_y, valid_x):
    cl.fit(training_x, training_y)
    preds = cl.predict(valid_x)
    return cl, preds, cl.score(training_x, training_y), cl.score(valid_x, valid_y), np.mean(preds)


def experiment(x, y):
    linear_x = [None, None, None]

    for i in xrange(3):
        linear_x[i] = make_samples_linear(x[i])

    scaler = sklearn.preprocessing.StandardScaler()
    linear_x[0] = scaler.fit_transform(linear_x[0])
    linear_x[1] = scaler.transform(linear_x[1])
    linear_x[2] = scaler.transform(linear_x[2])

    assert len(linear_x) == 3
    assert len(y) == 3

    training_x = linear_x[0]
    valid_x = linear_x[1]
    test_x = linear_x[2]

    training_y = y[0]
    valid_y = y[1]
    test_y = y[2]

    


    c_const = 10.
    arg_combinations = []
    classifier_futures = []
    print("STARTING")
    
    with futures.ProcessPoolExecutor(max_workers=int(subprocess.check_output("nproc")) // 2 - 1) as executor:
        for n_estimators_exp in range(1, 5):
            for max_features_exp in range(1, 20, 2):
                cl = RandomForestClassifier(n_estimators=10**n_estimators_exp, max_features=10**max_features_exp)
                classifier_futures.append(executor.submit(chain, cl, training_x, training_y, valid_x))
            
        print("--")
        print("Doing random forests based classification")
        for classifier_future in classifier_futures:
            cl, preds, tr_score, va_score, mean_preds = classifier_future.result()
            print("\t- classif obj:      {}".format(cl))
            print("\t- Training score:   {}".format(tr_score))
            print("\t- Valid score:      {}".format(va_score))
            print("\t- valid avg:        {}".format(mean_preds))
            print("\t--")

