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
import numpy as np

base_path = os.path.dirname(__file__)
default_summary_path = os.path.join(base_path, "saves", "tf_summaries")


class FFNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape_1, y_shape_1, depth, expected_minibatch_size, width_hidden_layers=2,
                 dropout_keep_prob=1.0, l2_c=0,
                 summary_writing_path=default_summary_path, activation_factory=NN_utils.relu_layer,):
        assert False, "NEEDS TO BE TESTED AGAIN BEFORE USE"
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

        self.finish_init(net, expected_minibatch_size)


class CNN(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, depth, dropout_keep_prob, filter_scale_factor, expected_minibatch_size,
                 first_layer_scale_factor=2, pool_every_k=3, summary_writing_path=default_summary_path):
        super(self.__class__, self).__init__(summary_writing_path)

        self.dropout_keep_prob = dropout_keep_prob
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        net = self._x
        for i in xrange(depth):
            input_depth = net.get_shape().as_list()[3]
            if i == 0:
                output_depth = input_depth * first_layer_scale_factor
            else:
                output_depth = int(input_depth * filter_scale_factor)

            filter_shape = (3, 3, input_depth, output_depth)
            net = NN_utils.conv_layer(net, filter_shape, 1)

            if i > pool_every_k:
                # VGG only starts dropout at the second 'conv N times then pool' block
                net = tf.nn.dropout(net, self._dropout_keep_prob)

            if pool_every_k is not None and (i + 1) % pool_every_k == 0 and i != 0:
                net = NN_utils.max_pool(net)

        self.finish_init(net, y_shape_1, expected_minibatch_size, x_shape, np.float32)

class VGG(NN_utils.AbstractClassifier):
    """
    VGG_small -> originally built for CIFAR
    """
    def __init__(self, x_shape, y_shape_1, dropout_keep_prob, expected_minibatch_size,
                 summary_writing_path=default_summary_path, ):
        super(self.__class__, self).__init__(summary_writing_path)
        glob_avg_pool = True

        self.dropout_keep_prob = dropout_keep_prob
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        def conv_adap(net, name, kh, kw, n_out, dh, dw):
            n_in = net.get_shape()[-1].value # this is the way we should also be doing it
            assert dh == dw
            return NN_utils.bn_conv_layer(net, (kh, kw, n_in, n_out), dh, )

        def mpool_adap(net, name, kh, kw, dh, dw):
            tests = (kh == 2, kw == 2, dh == 2, dw == 2) # kh == kw == dh == dw == 2 in pythonic magic probably
            assert all(tests), tests
            return NN_utils.max_pool(net, name)

        def fc_adap(input_op, name, n_out):
            n_in = input_op.get_shape()[-1].value

            with tf.name_scope(name) as scope:
                kernel = tf.get_variable(scope + "w",
                                         shape=[n_in, n_out],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
                activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
                return activation

        conv1_1 = conv_adap(self._x, name="conv1_1", kh=3, kw=3, n_out=16, dh=1, dw=1)
        conv1_2 = conv_adap(conv1_1, name="conv1_2", kh=3, kw=3, n_out=16, dh=1, dw=1)
        pool1 = mpool_adap(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

        """
        # block 2 -- outputs 56x56x128
        conv2_1 = tf.nn.dropout(conv_adap(pool1, name="conv2_1", kh=3, kw=3, n_out=32, dh=1, dw=1), dropout_keep_prob)
        conv2_2 = tf.nn.dropout(conv_adap(conv2_1, name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1), dropout_keep_prob)
        pool2 = mpool_adap(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)


        # block 3 -- outputs 28x28x256
        conv3_1 = tf.nn.dropout(conv_adap(pool2, name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1), dropout_keep_prob)
        conv3_2 = tf.nn.dropout(conv_adap(conv3_1, name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1), dropout_keep_prob)
        pool3 = mpool_adap(conv3_2, name="pool3", kh=2, kw=2, dh=2, dw=2)

        # block 4 -- outputs 14x14x512
        conv4_1 = tf.nn.dropout(conv_adap(pool3, name="conv4_1", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
        conv4_2 = tf.nn.dropout(conv_adap(conv4_1, name="conv4_2", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
        conv4_3 = tf.nn.dropout(conv_adap(conv4_2, name="conv4_3", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
        pool4 = mpool_adap(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)
        """
        if glob_avg_pool:
            top = tf.reduce_mean(pool1, [1, 2])
        """
        else:
            shp = pool4.get_shape()
            flattened_shape = shp[1].value * shp[2].value * shp[3].value # 128 * 14 * 14 = 25088, fine
            assert flattened_shape == 25088
            resh1 = tf.reshape(pool4, [-1, flattened_shape], name="resh1") # fine

            # fully connected
            fc6 = fc_adap(resh1, name="fc6", n_out=1024) # w should be 25088, 1024
            fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")

            fc7 = fc_adap(fc6_drop, name="fc7", n_out=1024) # w should be 1024, 1024
            top = tf.nn.dropout(fc7, dropout_keep_prob, name="fc7_drop")
        """
        self.finish_init(top, y_shape_1, expected_minibatch_size, x_shape, np.float32)


class ResNet(NN_utils.AbstractClassifier):
    def __init__(self, x_shape, y_shape_1, dropout_keep_prob, expected_minibatch_size,
                 summary_writing_path=default_summary_path):
        super(self.__class__, self).__init__(summary_writing_path)

        def easy_conv(net, n_out):
            n_in = net.get_shape()[-1].value  # this is the way we should also be doing it
            return NN_utils.bn_conv_layer(net, (3, 3, n_in, n_out), 1, lambda a: tf.maximum(0.2 * a, a))

        self.dropout_keep_prob = dropout_keep_prob
        self._x = tf.placeholder(tf.float32, shape=[None, x_shape[1], x_shape[2], x_shape[3]], name="x")
        self._y = tf.placeholder(tf.float32, shape=[None, y_shape_1], name="y")
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        net = self._x
        net = easy_conv(net, 16)
        NN_utils.residual_block(net, 16, True, non_lin=lambda a: tf.maximum(0.2 * a, a)) # leaky relu
        # NN_utils.residual_block(net, 32, True, non_lin=lambda a: tf.maximum(0.2 * a, a))
        top = tf.reduce_mean(net, [1, 2])

        self.finish_init(top, y_shape_1, expected_minibatch_size, x_shape, np.float32)
