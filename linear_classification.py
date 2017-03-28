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
from sklearn.svm import LinearSVC
from sklearn.linear_model import logistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing


def make_samples_linear(x):
    target_shape = (x.shape[utils.X_Dims.samples_and_times.value], x.shape[utils.X_Dims.fft_ch.value] * x.shape[utils.X_Dims.sensors.value])
    print("target_shape: {}".format(target_shape))
    linear_x = x.reshape(*target_shape)
    return linear_x


def linear_classification(x, y, job):
    linear_x = [None, None, None]

    print("Training shape:")

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

    classifiers = []

    print("Creating the classifiers")

    if "NN" in job:
        classifiers.append(
            nn_models.FFNN(
                x_shape_1=linear_x[0].shape,
                y_shape_1=2, depth=1, width_hidden_layers=10,
                dropout_keep_prob=0.5, l2_c=1))

    if "LSTM" in job:
        classifiers.append(
            nn_models.LSTM(
                x_shape=linear_x[0].shape,
                y_shape_1=2,
                seq_len=None,
                expected_minibatch_size=512))

    if "SVM" in job:
        c_const = 10.
        gamma_const = 10.

        for c_exp in xrange(-10, 10, 2):
            for gamma_exp in xrange(-10, 10, 2):
                classifiers.append(
                    SVC(C=c_const ** c_exp,
                        gamma=gamma_const ** gamma_exp,
                        kernel="linear",
                        #verbose=False,
                        verbose=True,
                        max_iter=300
                        ))

    if "rbf" in job:
        c_const = 10.
        gamma_const = 10.

        for c_exp in xrange(-10, 10, 3):
            for gamma_exp in xrange(-10, 10, 3):
                for degree in xrange(3, 7, 2):
                    classifiers.append(
                        SVC(C=c_const ** c_exp,
                            gamma=gamma_const ** gamma_exp,
                            degree=degree,
                            kernel="rbf",
                            #verbose=False,
                            verbose=True,
                            max_iter=300
                            ))

    if "KNN" in job:
        for metric in ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"]:
            for knn_exp in xrange(6):
                classifiers.append(KNeighborsClassifier(n_neighbors=2.**knn_exp, metric=metric))

    if "RandomForests" in job:
        for rf_n_estimators_exp in xrange(10):
            classifiers.append(RandomForestClassifier(n_estimators =2. ** rf_n_estimators_exp ))

    if "SKL_LR" in job:
        tol_const = 10.
        c_const = 10.
        for tol_exp in xrange(-5, -3, 1):
            for c_exp in xrange(-5, 1, 1):
                classifiers.append(logistic.LogisticRegression(max_iter=10000, verbose=10, tol=tol_const ** tol_exp,
                                                               C=c_const ** c_exp))

    assert len(classifiers) > 0, "No classifier to test."

    print("--")
    one_hot_set = {nn_models.FFNN}
    print("Doing linear classification")
    for classifier in classifiers:
        # If it's an sklearn classifier..
        if isinstance(classifier, sklearn.base.ClassifierMixin):
            cl = classifier.fit(training_x, training_y)
            # print("\t- Making predictions and calculating the accuracy scores")
            preds_1 = cl.predict(valid_x)
            print("\t- Training score:   {}".format(cl.score(training_x, training_y)))
            print("\t- Valid score:      {}".format(cl.score(valid_x, valid_y)))
            print("\t- valid avg:        {}".format(np.mean(preds_1)))
            print("\t- classif obj:      {}".format(cl))
            print("\t--")

        else:
            if type(classifier) in one_hot_set:
                y = [utils.to_one_hot(_y, 2) for _y in y[:2]]

            classifier.fit(
                train_x=linear_x[0],    train_y=y[0],
                valid_x=linear_x[1],    valid_y=y[1],
                n_epochs=1000000,       minibatch_size=1028,
                learning_rate=0.0001)


