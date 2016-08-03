# Compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

import numpy as np
import tensorflow as tf

import utils


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def softmax_layer(input_, shape):
    fc_w = weight_variable(shape)
    fc_b = tf.Variable(tf.truncated_normal([shape[1]]))

    fc_h = tf.nn.softmax(tf.matmul(input_, fc_w) + fc_b)

    return fc_h


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
    def fit(self, train_x, train_y, valid_x, valid_y, n_epochs, minibatch_size, learning_rate):
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

                if epoch % 500 == 0 and epoch != 0:
                    feed_dict = {self._x: valid_x[:, :], }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = 1.0
                    preds_va = sess.run([self.prediction], feed_dict=feed_dict)

                    feed_dict = {self._x: train_x[:, :], }
                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = 1.0
                    preds_tr = sess.run([self.prediction], feed_dict=feed_dict)

                    print("NN: epoch {}:".format(epoch))
                    print("\t- Loss: {}".format(loss))
                    print("\t- Score va: {:2.4f}".format(np.average(preds_va == utils.from_one_hot(valid_y))))
                    print("\t- Score tr: {:2.4f}".format(np.average(preds_tr == utils.from_one_hot(train_y))))
            print("--")

