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

default_base_path = os.path.join(os.path.dirname(__file__), "saves", "tf_summaries")
default_summary_path = default_base_path


class FFNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape_1, y_shape_1, depth, width_hidden_layers=2,
                 dropout_keep_prob=1.0, l2_c=0,
                 summary_writing_path=default_summary_path, activation_factory=NN_utils.relu_layer):

        super(self.__class__, self).__init__(summary_writing_path)
        self.dropout_keep_prob = dropout_keep_prob

        self._x = tf.placeholder(tf.float32, shape=[None, x_shape_1], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self._l2_c = l2_c

        net, (_, _) = activation_factory(self._x, [x_shape_1, width_hidden_layers])
        for _ in xrange(depth - 1):
            net, (_, _) = activation_factory(net, [width_hidden_layers, width_hidden_layers])
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        w_squares = sum([tf.reduce_sum(tf.matmul(x, tf.transpose(x))) for x in self._w_list])
        self._l2 = l2_c * w_squares

        self.finish_init(net)


class CNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, dropout_keep_prob, filter_scale_factor,
                 summary_writing_path=default_summary_path):
        super(self.__class__, self).__init__(summary_writing_path)
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob

        net = self._x
        for _ in xrange(depth):
            input_depth = net.get_shape().as_list()[3]
            output_depth = int(input_depth * filter_scale_factor)
            filter_shape = (3, 3, input_depth, output_depth)
            net = NN_utils.conv_layer(net, filter_shape, 1)
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        self.finish_init(net, y_shape_1)


class ResNet(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, dropout_keep_prob, filter_scale_factor,
                 summary_writing_path=default_summary_path):
        super(self.__class__, self).__init__(summary_writing_path)
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = dropout_keep_prob
        filter_scale_factor = tf.constant(filter_scale_factor, name="filter_scale_factor")

        net = self._x
        for _ in xrange(depth):
            input_depth = net.get_shape().as_list()[3]
            output_depth = int(input_depth * filter_scale_factor)
            net = NN_utils.residual_block(net, output_depth, False)
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        self.finish_init(net)


def make_interpolated_data(x, res, method, sample_info, sensor_type=True, show=False):
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

    interp_x = [None, None, None]

    # Merge grad data
    ## picks = mne.channels.layout._pair_grad_sensors(sample_info)
    ## x = mne.channels.layout._merge_grad_data(x[picks])

    for sample_set_idx in xrange(2): # There is currently no reason to do the test set. it being hardcoded is really poor,
                                     # but such is life
        interp_x[sample_set_idx] = np.empty([x[sample_set_idx].shape[utils.X_Dims.samples_and_times.value],
                                             x[sample_set_idx].shape[utils.X_Dims.fft_ch.value], res[0], res[1]],
                                            dtype=np.float32)
        sample_bound = x[sample_set_idx].shape[utils.X_Dims.samples_and_times.value]
        fft_bound = x[sample_set_idx].shape[utils.X_Dims.fft_ch.value]

        for sample_idx_per_set in xrange(sample_bound):
            if sample_idx_per_set % 5 == 0:
                sys.stdout.write("\rk: '{}', i: {} of {} ({:4.2f} %))".format(
                    sample_set_idx, sample_idx_per_set, sample_bound, 100 * sample_idx_per_set / sample_bound))
            for fft_channel_idx in xrange(fft_bound):
                psd_image = scipy.interpolate.griddata(
                    sensor_positions, x[sample_set_idx][sample_idx_per_set, fft_channel_idx, picks], grid, method, 0)
                interp_x[sample_set_idx][sample_idx_per_set, fft_channel_idx] = psd_image

                if show:
                    plt.imshow(psd_image, interpolation="none")
                    plt.show()

        assert np.all(np.isfinite(interp_x[sample_set_idx]))
        interp_x[sample_set_idx] = np.swapaxes(np.swapaxes(interp_x[sample_set_idx], 1, 2), 2, 3)

    return interp_x, str(abs((res, method, sensor_type).__hash__()))
