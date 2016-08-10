# Compatibility imports
from __future__ import with_statement, print_function, division

import os

import six
from six.moves import xrange
from six.moves import zip as izip

import numpy as np
import tensorflow as tf

import utils


def weight_variable(shape, name=None):
    initial = tf.random_normal(shape, stddev=0.5)
    return tf.Variable(initial, name=name)


def biais_variable(shape, name=None):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=name)


def softmax_layer(input_, shape):
    fc_w = weight_variable(shape)
    fc_b = biais_variable([shape[1]])
    fc_h = tf.nn.softmax(tf.matmul(input_, fc_w) + fc_b)

    return fc_h, (fc_w, fc_b)


def relu_layer(input_, shape):
    fc_w = weight_variable(shape)
    fc_b = biais_variable([shape[1]])
    fc_h = tf.nn.relu(tf.matmul(input_, fc_w) + fc_b)

    return fc_h, (fc_w, fc_b)


def conv_layer(input_, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(input_, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]), name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)

    out = tf.nn.relu(batch_norm)

    return out


def residual_block(input_, output_depth, down_sample, projection=False):
    input_depth = input_.get_shape().as_list()[3]

    if down_sample:
        filter_ = [1, 2, 2, 1]
        input_ = tf.nn.max_pool(input_, ksize=filter_, strides=filter_, padding='SAME')

    conv1 = conv_layer(input_, [3, 3, input_depth, output_depth], 1)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)

    if input_depth != output_depth:
        if projection:
            # Option B: Projection shortcut
            input_layer = conv_layer(input_, [1, 1, input_depth, output_depth], 2)
        else:
            # Option A: Zero-padding
            input_layer = tf.pad(input_, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        input_layer = input_

    res = conv2 + input_layer

    return res


class AbstractClassifier(object):
    def fit(self, train_x, train_y, valid_x, valid_y, n_epochs, minibatch_size, learning_rate, verbose=True):
        with tf.Session() as sess:
            sess.run([tf.initialize_all_variables()])
            for epoch in xrange(n_epochs):
                
                for start in xrange(0, train_x.shape[0], minibatch_size):
                    end = min(train_x.shape[0], start + minibatch_size)

                    feed_dict = {
                        self._x: train_x[start:end],
                        self._y: train_y[start:end],
                        self._lr: learning_rate,
                    }

                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = self.dropout_keep_prob
                    opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)

                if verbose:
                    feed_dict = {
                        self._x: train_x,
                        self._y: train_y
                    }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = 1.0
                    training_predictions, training_loss, training_score, training_l2 = sess.run([self.prediction, self.loss, self.score, self.l2], feed_dict=feed_dict)

                    feed_dict = {
                        self._x: valid_x,
                        self._y: valid_y
                    }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = 1.0
                    validation_predictions, validation_loss, validation_score, validation_l2 = sess.run([self.prediction, self.loss, self.score, self.l2], feed_dict=feed_dict)
                    os.system("tput reset")
                    print("NN: epoch {}:".format(epoch))
                    print("\t- validation loss:           {}".format(validation_loss))
                    print("\t- validation loss l2 ratio:  {}".format(validation_l2 / validation_loss))
                    print("\t- training loss:             {}".format(training_loss))
                    print("\t- training loss l2 ratio:    {}".format(training_l2 / training_loss))
                    print("\t- Score va:          {:2.4f}".format(np.average(validation_predictions == utils.from_one_hot(valid_y))))
                    print("\t- Score tr:          {:2.4f}".format(np.average(training_predictions == utils.from_one_hot(train_y))))
                    print("\t- lr:                {}".format(learning_rate))
                    print("\t- l2_c:              {}".format(vars(self).get("_l2_c", "N/A")))
                    print("\t- dropout_keep_prob: {}".format(vars(self).get("dropout_keep_prob", "N/A")))

                    training_res_counts = np.unique(training_predictions, return_counts=True)
                    if training_res_counts[0].shape[0] == 1:
                        print(">> There seems to only be one type of unique values in the training predictions. "
                              "This is almost certainly a bug.")

                    validation_res_counts = np.unique(validation_predictions, return_counts=True)
                    if validation_res_counts[0].shape[0] == 1:
                        print(">> There seems to only be one type of unique values in the valid predictions. "
                              "This is almost certainly a bug.")

                    print("\t- train prediction counts: {}".format(training_res_counts))
                    print("\t- valid prediction counts: {}".format(validation_res_counts))


                    for i, w in enumerate(vars(self).get("list_w", [])):
                        print("{}:\n{}".format(i, w))

                print("--")

