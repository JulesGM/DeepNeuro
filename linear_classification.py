# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

import utils
import NN_models

import numpy as np

# Sklearn imports
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import logistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def make_samples_linear(x):
    linear_x = x.reshape(x.shape[utils.X_Dims.samples_and_times.value], x.shape[utils.X_Dims.fft_ch.value] * x.shape[utils.X_Dims.sensors.value])
    return linear_x


def linear_classification(linear_x, linear_y, job):
    assert len(linear_x) == 3
    assert len(linear_y) == 3

    training_x = linear_x[0]
    valid_x = linear_x[1]
    test_x = linear_x[2]

    training_y = linear_y[0]
    valid_y = linear_y[1]
    test_y = linear_y[2]

    classifiers = []

    print("Creating the classifiers")

    if job == "NN":
        classifiers = [NN_models.FFNN(linear_x[0].shape, 2, 1, 12,)
                       ]
    elif job == "SVM":
        c_const = 10.
        for c_exp in xrange(-10, 19, 2):
            classifiers.append(SVC(C=c_const ** c_exp, kernel="linear", verbose=True, max_iter=1))

    elif job == "KNN":
        for metric in ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"]:
            for knn_exp in xrange(6):
                classifiers.append(KNeighborsClassifier(n_neighbors=2.**knn_exp, metric=metric))

    elif job == "RandomForests":
        for rf_n_estimators_exp in xrange(10):
            classifiers.append(RandomForestClassifier(n_estimators =2. ** rf_n_estimators_exp ))

    elif job == "SKL_LR":
        tol_const = 10.
        c_const = 10.
        for tol_exp in xrange(-5, -3, 1):
            for c_exp in xrange(-5, 1, 1):
                classifiers.append(logistic.LogisticRegression(max_iter=10000, verbose=10, tol=tol_const ** tol_exp,
                                                               C=c_const ** c_exp))

    print("--")
    one_hot_set = {NN_models.FFNN}
    print("Doing linear classification")

    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            one_hot_y = [utils.to_one_hot(_y, 2) for _y in linear_y[:2]]
            classifier.fit(linear_x[0], one_hot_y[0], linear_x[1], one_hot_y[1], 1000000, 64, 0.001)

        else:
            print("\t- Classifier:       {:30},   C={},  tol={}".format(
                    classifier.__class__, vars(classifier).get("C", "N/A"), vars(classifier).get("tol", "N/A")))

            print("\t- Fitting the model")
            cl = classifier.fit(training_x, training_y)
            assert cl is classifier
            print("\t- Making predictions and calculating the accuracy scores")
            preds_1 = cl.predict(valid_x)
            print("\t- Training score:   {}".format(cl.score(training_x, training_y)))
            print("\t- Valid score:      {}".format(cl.score(valid_x, valid_y)))
            print("\t- valid avg:        {}".format(np.mean(preds_1)))
            print("\t- classif obj:      {}".format(cl))

            print("\t--")


