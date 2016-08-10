#! /usr/bin/env python
from __future__ import print_function, division, with_statement
import os, sys, re, glob, argparse, fnmatch
import six
import numpy as np
import tensorflow as tf
import sklearn.datasets as skld

from collections import Counter

def to_one_hot(arr, N = None):
    if N is None:
        N = np.max(arr) + 1

    oh_arr = np.zeros((arr.shape[0], N))
    oh_arr[np.arange(arr.shape[0]), arr] = 1
    return oh_arr

def from_one_hot(oh_arr):
    return np.argmax(oh_arr, axis=1)

class AbstractClassifier(object):
    def fit(self, train_x, train_y, valid_x, valid_y, n_epochs, minibatch_size, learning_rate):
        with tf.Session() as sess:
            sess.run([tf.initialize_all_variables()])
            for epoch in six.moves.range(n_epochs):
                for start in range(0, train_x.shape[0], minibatch_size):
                    end = min(train_x.shape[0], start + minibatch_size)

                    feed_dict = {
                        self._x: train_x[start:end, :],
                        self._y: train_y[start:end],
                        self._lr: learning_rate,
                    }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = self.dropout_keep_prob
                    opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)

                if epoch % 500 == 0 and epoch != 0:
                    feed_dict = {self._x: valid_x[:, :], }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = 1.0
                    preds_va = sess.run([self.prediction], feed_dict=feed_dict)

                    feed_dict = {self._x: train_x[:, :], }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = 1.0
                    preds_tr = sess.run([self.prediction], feed_dict=feed_dict)

                    print("epoch {}:".format(epoch))
                    print("\t- Loss: {}".format(loss))
                    print("\t- Score va: {:2.4f}".format(np.average(preds_va == from_one_hot(valid_y))))
                    print("\t- Score tr: {:2.4f}".format(np.average(preds_tr == from_one_hot(train_y))))


class HighwayNN(AbstractClassifier):
    def __init__(self, x_shape_1, y_shape_0, depth, width, dropout):
        self.dropout_keep_prob = dropout

        self._x = tf.placeholder(tf.float32, shape=[None, x_shape_1])
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_0])
        self._lr = tf.placeholder(tf.float32)
        self._dropout_keep_prob = tf.placeholder(tf.float32)

        w0_s = [x_shape_1, x_shape_1]

        wn_s = [w0_s[1], w0_s[1]]
        bn_s = [w0_s[1]]

        w1_s = [w0_s[1], y_shape_0]
        b1_s = [y_shape_0]

        self._w = [None]
        self._b = [None]
        self._d = [self._x]

        N = depth
        for i in range(1, N):
            self._w.append(tf.Variable(initial_value=tf.truncated_normal(wn_s)))
            self._b.append(tf.Variable(initial_value=tf.truncated_normal(bn_s)))
            l = tf.nn.softmax(self._d[i - 1] + tf.matmul(self._d[i - 1], self._w[i]) + self._b[i])
            self._d.append(tf.nn.dropout(l, self._dropout_keep_prob))

        self._w1 = tf.Variable(initial_value=tf.truncated_normal(w1_s))
        self._b1 = tf.Variable(initial_value=tf.truncated_normal(b1_s))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(self._d[-1], self._w1) + self._b1, self._y))
        self.opt = tf.train.AdamOptimizer(self._lr).minimize(self.loss)
        self.prediction = tf.argmax(tf.nn.softmax(tf.matmul(self._d[-1], self._w1) + self._b1), 1)


class ResNet(AbstractClassifier):
    def __init__(self, x_shape_1, y_shape_0, depth, dropout, filter_dims, conv_strides, conv_paddings):
        self.dropout_keep_prob = dropout

        self._x = tf.placeholder(tf.float32, shape=[None, x_shape_1])
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_0])
        self._lr = tf.placeholder(tf.float32)
        self._dropout_keep_prob = tf.placeholder(tf.float32)

        self._side = tf.to_int32(np.sqrt(x_shape_1)) # this is a constant
        self._squared_x = tf.reshape(self._x, shape=tf.pack([-1, self._side, self._side, 1]))

        self._w = [None]
        self._b = [None]
        self._d = [self._squared_x]

        self._w.append(tf.Variable(initial_value=tf.truncated_normal(filter_dims[0])))
        self._b.append(tf.Variable(initial_value=tf.truncated_normal(filter_dims[0][1])))

        N = depth
        for i in range(1, N):
            self._w.append(tf.Variable(initial_value=tf.truncated_normal(filter_dims[i - 1])))
            self._b.append(tf.Variable(initial_value=tf.truncated_normal(filter_dims[i - 1][1])))
            conv = tf.nn.conv2d(self._d[i - 1], self._w[i], conv_strides[i - 1], conv_paddings[i - 1], use_cudnn_on_gpu=True)
            activ = tf.nn.relu(self._d[i - 1] + conv + self._b[i])
            dropout = tf.nn.dropout(activ, self._dropout_keep_prob)
            self._d.append(dropout)

        self._w1 = tf.Variable(initial_value=tf.truncated_normal(self._side, self._side, y_shape_0))
        self._b1 = tf.Variable(initial_value=tf.truncated_normal(y_shape_0))

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(tf.matmul(self._d[N - 1], self._w1) + self._b1, self._y))
        self.opt = tf.train.AdamOptimizer(self._lr).minimize(self.loss)
        self.prediction = tf.argmax(tf.nn.softmax(tf.matmul(self._d[N - 1], self._w1) + self._b1), 1)

def classify(dataset):
    iris = dataset
    print("\t- data.shape: {}".format(iris.data.shape))
    print("\t- target.shape: {}".format(iris.target.shape))

    x = iris.data
    y = to_one_hot(iris.target)
    idx = np.random.permutation(x.shape[0])

    valid_x = x[idx[:int(.2 * x.shape[0])], :]
    tr_x = x[idx[int(.2 * x.shape[0]):], :]

    valid_y = y[idx[:int(.2 * x.shape[0])], :]
    tr_y = y[idx[int(.2 * x.shape[0]):], :]

    #cl = HighwayNN(x.shape[1], y.shape[1], 2, 128, 0.5)
    #cl.fit(tr_x, tr_y, valid_x, valid_y, 10000, 64, 0.001)
    # def __init__(self, x_shape_1, y_shape_0, depth, dropout, filter_dims, conv_strides, conv_paddings):


    depth = 4
    conv_filters = [[1, 3, 3, 1]] * (depth - 1)
    conv_strides = [[1, 1, 1, 1]] * (depth - 1)
    conv_paddings = ["SAME"] * (depth - 1)
    cl = ResNet(x.shape[1], y.shape[1], depth, 1.0, conv_filters, conv_strides, conv_paddings)


def main(argv):
    TOLERATE_EXCEPTIONS = False
    #excluded_set = {"load_files", "load_sample_images"}
    skld_dict = vars(skld)
    good = {"load_boston", "load_iris", "load_diabetes", "load_digits", "load_linnerud"}

    for k in good:

        print("Doing {}".format(k.replace("load_", "")))
        if TOLERATE_EXCEPTIONS:
            try:
                classify(skld_dict[k])
            except Exception, err:
                print("Ignoring {} - {}".format(k, err))
                pass
        else:
            classify(skld_dict[k])

    classify(skld.load_iris())
    print("done")

if __name__ == "__main__" : sys.exit(main(sys.argv))
