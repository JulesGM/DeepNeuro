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


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.zeros([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h

def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out

def residual_block(inpt, output_depth, down_sample, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1,2,2,1]
        inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, output_depth - input_depth]])
    else:
        input_layer = inpt

    res = conv2 + input_layer

    return res

class AbstractClassifier(object):
    def fit(self, train_x, train_y, valid_x, valid_y, n_epochs, minibatch_size, learning_rate):
        with tf.Session() as sess:
            sess.run([tf.initialize_all_variables()])
            for epoch in six.moves.range(n_epochs):
                for start in range(0, train_x.shape[0], minibatch_size):
                    end = min(train_x.shape[0], start + minibatch_size)

                    feed_dict = {
                        self._x: train_x[start:end],
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


class ResNet(AbstractClassifier):

    def __init__(self, x_shape, y_shape_1, depth):

        self._x = tf.placeholder(tf.float32, shape=x_shape)
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1])
        self._lr = tf.placeholder(tf.float32)

        net = residual_block(self._x, x_shape[3], True)
        for i in range(depth):
            net = residual_block(net)

        self.score = softmax_layer(net, [None, y_shape_1])
        self.loss = tf.reduce_mean(-self._y * tf.log(self.score))
        self.opt = tf.train.AdamOptimizer(self._lr).minimize(self.loss)
        self.prediction = tf.argmax(self.score, 1)


