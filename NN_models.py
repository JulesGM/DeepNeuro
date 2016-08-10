from __future__ import print_function, generators, division, with_statement
from six import iteritems
from six.moves import zip as izip
from six.moves import range as xrange

import os
import sys

import numpy as np
import tensorflow as tf

import mne
import mne.time_frequency
import mne.channels.layout

import utils
import NN_utils
import custom_cells.tensorflow_resnet as tf_resnet
import custom_cells.tensorflow_resnet.resnet_train as tf_resnet_train


class FFNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, width_hidden_layers=2,
                 dropout_keep_prob=1.0, l2_c=0, activation_factory=NN_utils.relu_layer):
        self.dropout_keep_prob = dropout_keep_prob

        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self._l2_c = l2_c

        self.w_list = []
        self.b_list = []

        net, (w, b) = activation_factory(self._x, [x_shape[1], width_hidden_layers])
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
        b_squares = sum([tf.reduce_sum(x * x) for x in self.b_list])

        self.l2 = l2_c * w_squares
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a0, self._y)) + self.l2
        self.opt = tf.train.AdadeltaOptimizer(self._lr).minimize(self.loss)
        self.score = tf.nn.softmax(a0)
        self.prediction = tf.arg_max(self.score, 1)


class SmallResNet(NN_utils.AbstractClassifier):
    def __init__(self):
        self._is_training = tf.placeholder('bool', [], name='is_training')


        images, labels = tf.cond(self._is_training,           # if is_training
                                 lambda: (interp_x[0], Y[0]), # training
                                 lambda: (interp_x[1], Y[1]), # validation
                                )

        logits = tf_resnet.inference_small(images, num_classes=2, is_training=self._is_training, use_bias=False, num_blocks=2)
        tf_resnet_train.train(self._is_training, logits, images, labels)


def CNN(X, Y, sample_info, args):
    res_x = 10
    res_y = 10
    interp = "cubic"
    saver_loader = utils.data_utils.SaverLoader(
        "./interp_image_saves/{interp}_{resx}_{resy}_{limit}_{glob_tincr}_{nfft}_{name}_images.pkl" \
        .format(interp=interp, res_x=res_x, res_y=res_y, limit=args.limit, glob_tincr=args.glob_tincr,
                nfft=args.nfft, name="images"))

    if saver_loader.save_exists():
        interp_x = saver_loader.load_ds()

    else:
        interp_x = [None, None, None]
        for i in xrange(3):
            interp_x[i] = make_interpolated_data(X[i], (res_x, res_y), interp, sample_info)

        saver_loader.save_ds(interp_x)


def make_interpolated_data(X, res, method, sample_info, sensor_type=True, show=False):
    picks = mne.pick_types(sample_info, meg=sensor_type)
    sensor_positions = mne.channels.layout._auto_topomap_coords(sample_info, picks, True)

    # Take any valid file's position information, as all raws [are supposed to] have the same positions
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

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
            psd_image = griddata(sensor_positions[picks, :], X[i, j, picks], grid, method)
            interp_x[i, j, :] = psd_image[:, :]

            if show:
                plt.imshow(psd_image, interpolation="none")
                plt.show()

    return interp_x
