# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

import utils
import nn_models
import numpy as np

# Sklearn imports
from sklearn.svm import SVC
import sklearn.preprocessing

import subprocess
import concurrent.futures as futures


def make_samples_linear(x):
    target_shape = (x.shape[utils.X_Dims.samples_and_times.value], x.shape[utils.X_Dims.fft_ch.value] * x.shape[utils.X_Dims.sensors.value])
    print("target_shape: {}".format(target_shape))
    linear_x = x.reshape(*target_shape)
    return linear_x


def SVM(x, y, job):
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

    classifier_futures = []

    def chain(cl, training_x, training_y, valid_x):
        cl.fit(training_x, training_y)
        preds = cl.predict(valid_x)
        return cl, preds, cl.score(training_x, training_y), cl.score(valid_x, valid_y), np.mean(preds)

    c_const = 10.
    arg_combinations = []
    c_exp = 0.01
    print("STARTING")
    with futures.ThreadPoolExecutor(max_workers=int(subprocess.check_output("nproc")) - 1) as executor:
        for max_iter in range(11, 30, 2):
            cl = SVC(C=c_const ** c_exp,
                    kernel="linear",
                    #verbose=False,
                    # verbose=True, # http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html says that SVCs are somehow not thread safe when 
                    # verbose is on.
                    max_iter=max_iter * 100
                    )
            classifier_futures.append(executor.submit(chain, cl, training_x, training_y, valid_x))
                    
        print("--")
        print("Doing linear classification")
        for classifier_future in classifier_futures:
            cl, preds, tr_score, va_score, mean_preds = classifier_future.result()
            print("\t- classif obj:      {}".format(cl))
            print("\t- Training score:   {}".format(tr_score))
            print("\t- Valid score:      {}".format(va_score))
            print("\t- valid avg:        {}".format(mean_preds))
            print("\t--")

