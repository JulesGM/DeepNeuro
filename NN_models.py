# Compatibility
from __future__ import print_function, generators, division, with_statement
from six import iteritems
from six.moves import zip as izip
from six.moves import xrange

# Stdlib
import os

# Own
import NN_utils

# External
import tensorflow as tf

base_path = os.path.dirname(__file__)
default_summary_path = os.path.join(base_path, "saves", "tf_summaries")


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

        net = activation_factory(self._x, [x_shape_1, width_hidden_layers])
        for _ in xrange(depth - 1):
            net = activation_factory(net, [width_hidden_layers, width_hidden_layers])
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        w_squares = sum([tf.reduce_sum(tf.matmul(x, tf.transpose(x))) for x in self._w_list])
        self._l2 = l2_c * w_squares

        self.finish_init(net)


class CNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, dropout_keep_prob, filter_scale_factor,
                 summary_writing_path=default_summary_path):
        super(self.__class__, self).__init__(summary_writing_path)

        self.dropout_keep_prob = dropout_keep_prob
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        filter_scale_factor = tf.constant(filter_scale_factor, name="filter_scale_factor")

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

        self.dropout_keep_prob = dropout_keep_prob
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        filter_scale_factor = tf.constant(filter_scale_factor, name="filter_scale_factor")

        net = self._x
        for _ in xrange(depth):
            input_depth = net.get_shape().as_list()[3]
            output_depth = int(input_depth * filter_scale_factor)
            net = NN_utils.residual_block(net, output_depth, False)
            net = tf.nn.dropout(net, self._dropout_keep_prob)

        self.finish_init(net)
