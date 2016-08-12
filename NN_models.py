from __future__ import print_function, generators, division, with_statement
from six import iteritems
from six.moves import zip as izip
from six.moves import xrange

import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.interpolate

import mne
import mne.time_frequency
import mne.channels.layout

import utils
import NN_utils
import custom_cells.tensorflow_resnet as tf_resnet
import custom_cells.tensorflow_resnet.resnet_train as tf_resnet_train


class FFNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape_1, y_shape_1, depth, width_hidden_layers=2,
                 dropout_keep_prob=1.0, l2_c=0, activation_factory=NN_utils.relu_layer):
        self.dropout_keep_prob = dropout_keep_prob

        self._x = tf.placeholder(tf.float32, shape=[None, x_shape_1], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self._l2_c = l2_c

        self.w_list = []
        self.b_list = []

        net, (w, b) = activation_factory(self._x, [x_shape_1, width_hidden_layers])
        self.w_list.append(w)
        self.b_list.append(b)

        for i in xrange(depth - 1):
            net, (w, b) = activation_factory(net, [width_hidden_layers, width_hidden_layers])
            net = tf.nn.dropout(net, self._dropout_keep_prob)
            self.w_list.append(w)
            self.b_list.append(b)

        w0 = tf.Variable(tf.truncated_normal([width_hidden_layers, y_shape_1]))
        b0 = tf.Variable(tf.truncated_normal([y_shape_1]))
        a0 = tf.matmul(net, w0) + b0

        self.w_list.append(w0)
        self.b_list.append(b0)

        w_squares = sum([tf.reduce_sum(tf.matmul(x, tf.transpose(x))) for x in self.w_list])
        # b_squares = sum([tf.reduce_sum(x * x) for x in self.b_list])

        self.l2 = l2_c * w_squares
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a0, self._y)) + self.l2
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)
        self.score = tf.nn.softmax(a0)
        self.prediction = tf.arg_max(self.score, 1)


class SmallResNet(object):
    def __init__(self, training_interp_x, training_y, validation_interp_x, validation_y, depth, num_classes, learning_rate, minibatch_size):
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate

        self._is_training = tf.placeholder('bool', [], name='is_training')

        self._images, self._labels = tf.cond(self._is_training,
                                 lambda: (training_interp_x,   training_y),
                                 lambda: (validation_interp_x, validation_y),)

        self._logits = tf_resnet.inference_small(self._images, is_training=self._is_training,
                                                 use_bias=False, num_blocks=depth, num_classes=num_classes)

    def fit(self):
        tf_resnet_train.train(self._is_training, self._logits, self._images, self._labels, self.minibatch_size,
                              self.learning_rate)


class CNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, dropout_keep_prob):
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob


        net = NN_utils.conv_layer(self._x, (3, 3, x_shape[3], x_shape[3] * 2), 1)
        net = tf.nn.dropout(net, self._dropout_keep_prob)
        for i in xrange(depth - 1):
            net = NN_utils.conv_layer(net, (3, 3, net.get_shape().as_list()[3], net.get_shape().as_list()[3] * 2), 1)
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        shape = net.get_shape().as_list()
        w0 = tf.Variable(tf.truncated_normal([np.product(net.get_shape().as_list()[1:]), y_shape_1]))
        b0 = tf.Variable(tf.truncated_normal([y_shape_1]))
        a0 = tf.matmul(tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]]), w0) + b0

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a0, self._y))
        self.opt = tf.train.AdamOptimizer(self._lr).minimize(self.loss)
        self.score = tf.nn.softmax(a0)
        self.prediction = tf.arg_max(self.score, 1)


class BetterResNet(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, dropout_keep_prob):
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob


        net = NN_utils.residual_block(self._x, x_shape[3] * 2, False)
        net = tf.nn.dropout(net, self._dropout_keep_prob)
        for i in xrange(depth - 1):
            net = NN_utils.residual_block(net, net.get_shape().as_list()[3] * 2, False)
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        shape = net.get_shape().as_list()
        w0 = tf.Variable(tf.truncated_normal([np.product(net.get_shape().as_list()[1:]), y_shape_1]))
        b0 = tf.Variable(tf.truncated_normal([y_shape_1]))
        a0 = tf.matmul(tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]]), w0) + b0

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a0, self._y))
        self.opt = tf.train.AdamOptimizer(self._lr).minimize(self.loss)
        self.score = tf.nn.softmax(a0)
        self.prediction = tf.arg_max(self.score, 1)



def make_interpolated_data(X, res, method, sample_info, sensor_type=True, show=True):
    picks = mne.pick_types(sample_info, meg=sensor_type)
    sensor_positions = mne.channels.layout._auto_topomap_coords(sample_info, picks, True)

    # Take any valid file's position information, as all raws [are supposed to] have the same positions

    assert len(sensor_positions.shape) == 2 and sensor_positions.shape[1] == 2, sensor_positions.shape[1]
    min_x = np.floor(np.min(sensor_positions[:, 0]))
    max_x = np.ceil(np.max(sensor_positions[:, 0]))
    min_y = np.floor(np.min(sensor_positions[:, 1]))
    max_y = np.ceil(np.max(sensor_positions[:, 1]))

    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, res[0],), np.linspace(min_y, max_y, res[1]))
    grid = (grid_x, grid_y)

    interp_x = np.empty([X.shape[utils.X_Dims.samples_and_times.value], X.shape[utils.X_Dims.fft_ch.value], res[0], res[1]], dtype=np.float32)

    i_bound = X.shape[utils.X_Dims.samples_and_times.value]
    j_bound = X.shape[utils.X_Dims.fft_ch.value]
    for i in xrange(i_bound):
        if i % 5 == 0:
            sys.stdout.write("\ri: {} of {} ({:4.2f} %))".format(i, i_bound, 100 * i / i_bound))
        for j in xrange(j_bound):
            psd_image = scipy.interpolate.griddata(sensor_positions[picks, :], X[i, j, picks], grid, method)
            interp_x[i, j, :] = psd_image[:, :]

            if show:
                plt.imshow(psd_image, interpolation="none")
                plt.show()

    return interp_x
