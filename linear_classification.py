#! /usr/bin/env python

# Compatibility imports
from __future__ import with_statement, print_function, division
from six import iteritems
from six.moves import zip as izip

# Stdlib imports
import sys, os, argparse, time, logging, enum, json, subprocess as sp

# scipy/numpy/matplotlib/tf
import numpy as np
import tensorflow as tf

# Sklearn imports
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import logistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Varia
import tflearn
import h5py

from utils import *


def make_samples_linear(X):
    linear_X = X.reshape(X.shape[X_Dims.samples_and_times.value],  X.shape[X_Dims.fft_ch.value] * X.shape[X_Dims.sensors.value])
    return linear_X


class AbstractTensorflowLinearClassifier(object):
    def fit(self, linear_x, linear_y, n_epoch, minibatch_size=64):
        for i in xrange(3):
            std_x = np.std(linear_x[i])
            assert std_x > 1E-3, std_x
            std_y = np.std(linear_y[i])
            assert std_y > 1E-3, std_y

        assert type(minibatch_size) == int, "minibatch_size should be an integer. it's currently an {}".format(
            type(n_epoch))
        assert minibatch_size > 0, "n_epoch should be larger than zero. its value is currently '{}'".format(n_epoch)
        is_power_of_2 = minibatch_size & (minibatch_size - 1) == 0
        assert is_power_of_2, "minibatch_size should be an integer power of 2. It's log2 is currently '{}', with " \
                              "it's value being '{}'.".format(np.log2(minibatch_size), minibatch_size)

        training_x = linear_x[0]
        training_y = linear_y[0]

        set_names = ["scores_training", "scores_valid"]

        with tf.Session() as s:
            s.run([tf.initialize_all_variables()])
            for i in xrange(n_epoch):
                for j in xrange(0, training_x.shape[0] // minibatch_size + 1):
                    idx_from = j * minibatch_size
                    idx_to = min((j + 1) * minibatch_size, training_x.shape[0] - 1)
                    diff = idx_to - idx_from

                    if diff == 0:
                        print("diff == 0, skipping")
                        break

                    feed_dict = {self.x_ph: training_x[idx_from:idx_to, :],
                                 self.y_ph: training_y[idx_from:idx_to, :],
                                 self.dropout_keep_prob: self._dropout_keep_prob,
                                 }

                    loss, opt = s.run([self.loss, self.opt], feed_dict=feed_dict)

                # Save both the training and validation score to hdf5
                if i % 100 == 0 and i != 0:
                    print("EPOCH {}".format(i))
                    sys.stdout.write(">>> {_type}::{epoch}:: LOSS {loss}\n".format(_type=self._type, epoch=i, loss=loss))
                    for set_id in xrange(2):
                        feed_dict = {self.x_ph: linear_x[set_id][:, :],
                                     self.dropout_keep_prob: 1.0,
                                     }

                        preds = s.run(self.classif, feed_dict=feed_dict)
                        #print("preds")
                        #print(preds.shape)
                        #print(preds)
                        decision = np.argmax(preds, axis=1)

                        print("argmax preds")
                        #print(decision.shape)
                        print(decision)
                        label = np.argmax(linear_y[set_id], axis=1)

                        #print("argmax label")
                        #print(label.shape)
                        #print(label)
                        score = np.mean(label == decision)
                        set_name = set_names[set_id]
                        sys.stdout.write("{_type}::{epoch}::{set_name}: {score}\n".format(
                                _type=self._type, set_name=set_name, epoch=i, score=score))

                        if i % 1000 == 0 and i != 0:
                            print("FLUSHING")
                            sys.stdout.flush()


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
    job = "SVM"

    print("Creating the classifiers")
    from NN import FFNN

    if job == "NN":
        classifiers = [FFNN(linear_x[0].shape, 2, 2, 100)
                       ]
    elif job == "SVM":
        c_const = 10.
        for c_exp in range(-10, 19, 2):
            classifiers.append(SVC(C=c_const ** c_exp, kernel="linear", verbose=True, max_iter=1))

    elif job == "KNN":
        for metric in ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"]:
            for knn_exp in range(0, 6):
                classifiers.append(KNeighborsClassifier(n_neighbors=2.**knn_exp, metric=metric))

    elif job == "RandomForests":
        for rf_n_estimators_exp in range(0, 10):
            classifiers.append(RandomForestClassifier(n_estimators =2. ** rf_n_estimators_exp ))

    elif job == "SKL_LR":
        tol_const = 10.
        C_const = 10.
        for tol_exp in range(-5, -3, 1):
            for C_exp in range(-5, 1, 1):
                classifiers.append(logistic.LogisticRegression(max_iter=10000, verbose=10, tol=tol_const ** tol_exp, C=C_const ** C_exp))

    print("--")
    one_hot_set = {FFNN}
    print("Linearly classifying")
    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            one_hot_y = [to_one_hot(_y, 2) for _y in linear_y[:2]]
            # def fit(self, train_x, train_y, valid_x, valid_y, n_epochs, minibatch_size, learning_rate):
            classifier.fit(linear_x[0], one_hot_y[0], linear_x[1], one_hot_y[1], 1000000, 64, 0.01)
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


