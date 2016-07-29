#! /usr/bin/env python

# Compatibility imports
from __future__ import with_statement, print_function, division
from six import iteritems
from six.moves import zip as izip

# Stdlib imports
import sys, os, argparse, time, logging, enum, json, subprocess as sp

from utils.data_utils import *

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


def make_samples_linear(X, Y):
    linear_X = X.reshape(X.shape[X_Dims.samples_and_times.value],  X.shape[X_Dims.fft_ch.value] * X.shape[X_Dims.sensors.value])
    return linear_X, Y


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

        hdf5_file_name = "{}_{}.h5".format(self._type, time.time())
        hdf5_path = os.path.join(os.path.dirname(__file__), "scores", hdf5_file_name)
        output = h5py.File(hdf5_path, "a", libver='latest', compression=None)

        set_names = ["scores_training", "scores_valid"]
        output.create_dataset(set_names[0], data=np.nan * np.ones((n_epoch,), np.float32))
        output.create_dataset(set_names[1], data=np.nan * np.ones((n_epoch,), np.float32))

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
                        output[set_name][i] = score
                        sys.stdout.write("{_type}::{epoch}::{set_name}: {score}\n".format(
                                _type=self._type, set_name=set_name, epoch=i, score=score))

                        if i % 1000 == 0 and i != 0:
                            print("FLUSHING")
                            output.flush()
                            sys.stdout.flush()


class LogReg(AbstractTensorflowLinearClassifier):
    def __init__(self, learning_rate, input_ph_shape, output_ph_shape):
        self._type = "LR"

        self.x_ph = tf.placeholder(dtype=tf.float32, shape=input_ph_shape)
        self.y_ph = tf.placeholder(dtype=tf.float32, shape=output_ph_shape)

        w0_s = (input_ph_shape[1], output_ph_shape[1])
        b0_s = (output_ph_shape[1],)

        self.w0 = tf.Variable(initial_value=tf.truncated_normal(w0_s), dtype=tf.float32)
        self.b0 = tf.Variable(initial_value=tf.truncated_normal(b0_s), dtype=tf.float32)

        self.classif = tf.nn.softmax(tf.matmul(self.x_ph, self.w0) + self.b0)
        self.loss = tf.reduce_mean(- tf.reduce_sum(self.y_ph * tf.log(self.classif), reduction_indices=[1])) \
                    + 0.6 * tf.nn.l2_loss(self.w0)

        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



class FFNN(AbstractTensorflowLinearClassifier):
    def __init__(self, learning_rate, dropout_keep_prob, l2_c, input_ph_shape, output_ph_shape):
        self._type = "NN"
        self._dropout_keep_prob = dropout_keep_pro
        no_of_hidden_units = 10


        self.x_ph = tf.placeholder(dtype=tf.float32, shape=input_ph_shape)
        self.y_ph = tf.placeholder(dtype=tf.float32, shape=output_ph_shape)
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)

        w0_s = (input_ph_shape[1], no_of_hidden_units)
        b0_s = (w0_s[1],)
        w1_s = (w0_s[1], output_ph_shape[1])
        b1_s = (output_ph_shape[1],)

        self.w0 = tf.Variable(initial_value=tf.truncated_normal(w0_s), dtype=tf.float32)
        self.b0 = tf.Variable(initial_value=tf.truncated_normal(b0_s), dtype=tf.float32)
        self.w1 = tf.Variable(initial_value=tf.truncated_normal(w1_s), dtype=tf.float32)
        self.b1 = tf.Variable(initial_value=tf.truncated_normal(b1_s), dtype=tf.float32)

        self.l0 =      tf.nn.softmax(tf.matmul(self.x_ph, self.w0) + self.b0)
        self.d0 =      tf.nn.dropout(self.l0, self.dropout_keep_prob)
        self.classif = tf.nn.softmax(tf.matmul(self.l0,   self.w1) + self.b1)
        self.loss = tf.reduce_mean(- tf.reduce_sum(self.y_ph * tf.log(self.classif), reduction_indices=[1])) + \
                    l2_c * tf.nn.l2_loss(self.w0) + l2_c * tf.nn.l2_loss(self.w1)

        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



def linear_classification(linear_x, linear_y, job):
    header = ("*********************************************************\n"
              "**** Classification Code :                               \n"
              "*********************************************************")
    print(header)
    sys.stderr.write(header + "\n")
    sys.stderr.flush()

    assert len(linear_x) == 3
    assert len(linear_y) == 3

    feature_width = linear_x[0].shape[1]

    training_x = linear_x[0]
    valid_x = linear_x[1]
    test_x = linear_x[2]

    training_y = linear_y[0]
    valid_y = linear_y[1]
    test_y = linear_y[2]

    classifiers = []

    if job == "LR":
        classifiers = [
                        LogReg(0.00001, [None, feature_width], [None, 2]),
                        logistic.LogisticRegression(),
                       ]

    elif job == "NN":
        classifiers = [
                        FFNN(0.0001, 1.0, 0.3, [None, feature_width], [None, 2]),
                       ]

    elif job == "SVM":
        # svm grid search
        for tol_exp in range(-15, -5, 1):
            for c_exp in range(1, 5, 1):
                classifiers.append(LinearSVC(tol=10.**tol_exp, C=10.**c_exp))

    elif job == "KNN":
        for metric in ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"]:
            for knn_exp in range(0, 6):
                classifiers.append(KNeighborsClassifier(n_neighbors=2.**knn_exp, metric=metric))

    elif job == "RandomForests":
        for rf_n_estimators_exp in range(0, 10):
            classifiers.append(RandomForestClassifier(n_estimators =2. ** rf_n_estimators_exp ))


        """

    one_hot_set = {tflearn.DNN, LogReg, FFNN}
    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            one_hot_y = [to_one_hot(_y, 2) for _y in linear_y]
            classifier.fit(linear_x, one_hot_y, n_epoch=300000) #, validation_set=(features_va, labels_va))

            if type(classifier) == tflearn.DNN:
                predicted_valid_y = np.argmax(classifier.predict(valid_x), axis=1)
                predicted_train_y = np.argmalx(classifier.predict(training_x), axis=1)

                print("-------------------------------------")
                print("classifier:     {}".format(classifier))
                print("training score: {}".format(np.mean(predicted_train_y == one_hot_y[0])))
                print("valid score:    {}".format(np.mean(predicted_valid_y == one_hot_y[1])))
                print("-------------------------------------")

        else:
            labels_tr = training_y
            cl = classifier.fit(training_x, labels_tr)
            print("-")
            print("classifier:       {}".ljust(30, " ").format(classifier.__class__) + "   C={},  tol={}".format( vars(classifier).get("C", "N/A"),
                                                            vars(classifier).get("tol", "N/A")))
            print("training score:   {}".format(cl.score(training_x, training_y)))
            print("valid score:      {}".format(cl.score(valid_x, valid_y)))
            """


