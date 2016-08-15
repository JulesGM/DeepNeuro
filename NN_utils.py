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
    with tf.name_scope("weights"):
        initial = tf.random_normal(shape, stddev=0.5)
        var = tf.Variable(initial, name=name)
    return var


def biais_variable(shape, name=None):
    with tf.name_scope("bias"):
        initial = tf.zeros(shape)
    return tf.Variable(initial, name=name)


def softmax_layer(input_, shape):
    with tf.name_scope("softmax_layer"):
        fc_w = weight_variable(shape)
        fc_b = biais_variable([shape[1]])
        fc_h = tf.nn.softmax(tf.matmul(input_, fc_w) + fc_b)
    return fc_h, (fc_w, fc_b)


def relu_layer(input_, shape):
    with tf.name_scope("relu_layer"):
        fc_w = weight_variable(shape)
        fc_b = biais_variable([shape[1]])
        fc_h = tf.nn.relu(tf.matmul(input_, fc_w) + fc_b)
    return fc_h, (fc_w, fc_b)


def conv_layer(input_, filter_shape, stride):
    with tf.name_scope("conv_layer"):
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
    with tf.name_scope("residual_block"):
        input_depth = input_.get_shape().as_list()[3]

        if down_sample:
            filter_ = [1, 2, 2, 1]
            input_ = tf.nn.max_pool(input_, ksize=filter_, strides=filter_, padding='SAME')

        filters = []
        conv1 = conv_layer(input_, [3, 3, input_depth, output_depth], 1)
        filters.append(filter)

        conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1)
        filters.append(filter)

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
    def __init__(self, summary_writing_path):
        self._loss = None
        self._opt = None
        self._score = None
        self._prediction = None
        self._merged = None
        self._cross_valid_class = None
        self._lr = None
        self._x = None
        self._y = None
        self._accuracy = None
        self.summary_writing_path = summary_writing_path

    def finish_init(self, net, y_shape_1):
        # it's not clear that this shouldn't be in each of the little models. it has the advantage of reducing
        # the test surface, which is a huge advantage for a short term, precision reliant project

        self._cross_valid_class = tf.placeholder(dtype=tf.string)

        shape = net.get_shape().as_list()
        with tf.name_scope("FC_top"):
            w0 = tf.Variable(tf.truncated_normal([np.product(net.get_shape().as_list()[1:]), y_shape_1]), name="weights")
            b0 = tf.Variable(tf.truncated_normal([y_shape_1]), name="bias")
            a0 = tf.matmul(tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]]), w0) + b0

        with tf.name_scope("loss"):
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(a0, self._y))

            if "l2" in vars(self):
                self._loss += self.l2

            tf.scalar_summary("loss/" + self._cross_valid_class, self._loss)

        self._opt = tf.train.AdamOptimizer(self._lr).minimize(self._loss)
        self._score = tf.nn.softmax(a0)
        self._prediction = tf.arg_max(self._score, 1)
        self._merged = tf.merge_all_summaries()

        with tf.name_scope("accuracy"):
            self._accuracy = tf.reduce_mean(tf.cast(self._prediction == tf.argmax(self._y, 1), tf.float32))
            tf.scalar_summary('accuracy/' + self._cross_valid_class, self._accuracy)

    def _update_summaries(self, sess, x, y, cv_set, epoch, writer, verbose):
        """
        Only ran from AbstractClassifier.fit. Basically, the logging done for each cross-validation set
        """
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        feed_dict = {
            self._x: x,
            self._y: y,
            self._cross_valid_class: cv_set
        }
        if "_dropout_keep_prob" in vars(self):
            feed_dict[self._dropout_keep_prob] = 1.0

        maybe_l2 = vars(self).get("_l2", tf.constant(0))
        merged, accuracy, loss, predictions, l2 = sess.run([
            self._merged, self._accuracy, self._loss, self._prediction, maybe_l2],
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata,
        )

        writer.add_summary(merged, epoch)
        writer.add_run_metadata(run_metadata, 'step%d' % epoch)

        if verbose:
            print("\t{} summary:".format(cv_set))
            print("\t- {} accuracy:  {}".format(cv_set, accuracy))
            print("\t- {} loss:      {}".format(cv_set, loss))
            print("\t- {} l2:        {}".format(cv_set, l2))
            res_counts = np.unique(predictions, return_counts=True)
            if res_counts[0].shape[0] == 1:
                print(">> There seems to only be one type of unique values in the {} predictions.".format(cv_set))
            print("\t- {} prediction counts: {}".format(cv_set, res_counts))

    def fit(self, train_x, train_y, valid_x, valid_y, n_epochs, minibatch_size, learning_rate, verbose=True):
        with tf.Session() as sess:
            training_writer = tf.train.SummaryWriter(os.path.dirname(os.path.dirname(__file__)) + "/tf_summaries/valid")
            validation_writer = tf.train.SummaryWriter("../tf_summaries/valid")

            sess.run([tf.initialize_all_variables()])
            for epoch in xrange(n_epochs):
                
                for start in xrange(0, train_x.shape[0], minibatch_size):
                    end = min(train_x.shape[0], start + minibatch_size)

                    feed_dict = {
                        self._x:  train_x[start:end],
                        self._y:  train_y[start:end],
                        self._lr: learning_rate,
                        self._cross_valid_class: "training",}

                    if "_dropout_keep_prob" in vars(self):
                        feed_dict[self._dropout_keep_prob] = self.dropout_keep_prob
                    sess.run([self._opt, self._loss], feed_dict=feed_dict)

                ###############################
                # Summary writing section
                ###############################
                if verbose:
                    print("NN: epoch {}:".format(epoch))
                    print("\t- l2_c:              {}".format(vars(self).get("_l2_c", "N/A")))
                    print("\t- dropout_keep_prob: {}".format(vars(self).get("dropout_keep_prob", "N/A")))
                    print("")
                self._update_summaries(sess, train_x, train_y, "training",   epoch, training_writer,   verbose)
                self._update_summaries(sess, valid_x, valid_y, "validation", epoch, validation_writer, verbose)
                if verbose:
                    print("--")

