# Compatibility
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

# Stdlib
import os
import sys
import random
import time

# Own
import utils
import utils.data_utils
import nn_models

# External
import tflearn
import tensorflow as tf
import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.layers import Conv2D, Flatten, Dense, normalization


base_path = os.path.dirname(os.path.realpath(__file__))


def _make_image_save_name(is_time_dependant, res, sensor_type, nfft, fmax, tincr, use_established_bands):
    # we right them all, on purpose, instead of using *args, to make sure everything is in its place
    res_str = "_".join([str(d) for d in res])
    args = [is_time_dependant, res_str, sensor_type, nfft, fmax, tincr, use_established_bands]
    return "_".join([str(x) for x in args])

from tflearn.metrics import Metric
class tflearn_Prediction_Counts(Metric):
    """ Prints the count of each category of prediction that is present in the predictions.
    Can be useful to see, for example, to see if the model only gives one type of predictions,
    or if the predictions given are in the expected proportions """

    def __init__(self, inner_metric, name=None):
        super(tflearn_Prediction_Counts, self).__init__(name)
        self.inner_metric = inner_metric

    def build(self, predictions, targets, inputs=None):
        """ Prints the number of each kind of prediction """
        self.built = True
        pshape = predictions.get_shape()
        self.inner_metric.build(predictions, targets, inputs)

        with tf.name_scope(self.name):
            if len(pshape) == 1 or (len(pshape) == 2 and int(pshape[1]) == 1):
                self.name = self.name or "binary_prediction_counts"
                y, idx, count = tf.unique_with_counts(tf.argmax(predictions))
                self.tensor = tf.Print(self.inner_metric, [y, count, ], name=self.inner_metric.name)

            else:
                self.name = self.name or "categorical_prediction_counts"
                # y, idx, count = tf.unique_with_counts(tf.argmax(predictions, dimension=1))
                self.tensor = self.inner_metric.tensor

prediction_counts = tflearn_Prediction_Counts


def gen_summ_name(model_name, image_save_name):
    random_extension_string = str(random.randint(0, 1000000000000))
    summary_path = os.path.join(base_path, "saves", "tf_summaries", model_name + "_" + image_save_name + "_" +
                                random_extension_string)
    return summary_path


def spatial_classification(args):
    saves_loc = os.path.join(base_path, "saves/interp_image_saves")
    if not os.path.exists(saves_loc):
        os.mkdir(saves_loc)

    image_name = _make_image_save_name(args.is_time_dependant, args.res, args.sensor_type, args.nfft, args.tincr, args.fmax, args.established_bands)
    image_save_name = image_name + ".h5"
    hdf5_saver_loader = utils.data_utils.HDF5SaverLoader(os.path.join(saves_loc, image_save_name))

    if hdf5_saver_loader.save_exists():
        print("--")
        print("Unpickling {}.".format(hdf5_saver_loader.save_path))
        prepared_x = hdf5_saver_loader.load_ds()
        print("Done unpickling.")
        print("--")

    else:
        start = time.time()
        utils.data_utils.make_interpolated_data(x=args.x, res=args.res, method="cubic", info=args.info,
                                            sensor_type=args.sensor_type, show=False,
                                            hdf5_saver_loader=hdf5_saver_loader)

        prepared_x = hdf5_saver_loader.load_ds()
        sys.stderr.write("\n\n")
        print("\nPreparation of the images took {} seconds".format(time.time() - start))
        print("Shape of the prepared dataset {}".format(" & ".join([str(prep_x.shape) for prep_x in prepared_x[:2]])))

        # We want to use the mem mapped hdf5 instead of keeping our tens of gigs of images in ram,
        # save_ds also loads the hdf5 file it saved to.
        # Ideally, we would write the images to the files as they get computed. This is a TODO.

    for i in xrange(2):
        args.y[i] = utils.to_one_hot(args.y[i], np.max(args.y[i]) + 1)

    train_data = [prepared_x[0], args.y[0]]
    valid_data = [prepared_x[1], args.y[1]]

    train_mean = np.mean(train_data[0])
    train_std = np.std(train_data[0])

    train_data[0] = (train_data[0] - train_mean) / train_std
    valid_data[0] = (valid_data[0] - train_mean) / train_std

    x_shape = train_data[0].shape
    y_shape_1 = train_data[1].shape[1]
    input_shape = list(train_data[0].shape[1:])

    time_steps = 9009090
    sequence_length = 10
    sequence_stride = 5
    mass_concat = False

    if args.net_type == "lstm":
        x = tf.placeholder(np.float32, shape=[None] + [time_steps] + input_shape)
        y = tf.placeholder(np.float32, shape=[None] + [2])

        c0_depth = 32

        lstm = tf.nn.rnn_cell.BasicLSTMCell(c0_depth)
        state = tf.zeros(c0_depth)
        lstm_output = []

        for i in xrange(0, sequence_length, sequence_stride):
            net = tflearn.conv_2d(x[:, i, :, :, :], c0_depth, 3, activation="relu",
                                  weights_init=tf.contrib.layers.variance_scaling_initializer())
            net = tflearn.batch_normalization(net)
            net = tflearn.global_avg_pool(net)
            net = tflearn.flatten(net)
            net = tflearn.batch_normalization(net)
            net, state = lstm(net, state)

            lstm_output.append(net)

        if mass_concat:
            net = tf.concat(1, lstm_output)

        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(net, 2, activation="linear")

        # Training
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, y), reduction_indices=[1])
        opt = tf.train.AdamOptimizer().minimize(loss)

        # Metrics
        pred = tf.argmax(tf.nn.softmax(net), dimension=1)
        label = tf.argmax(y, dimension=1)
        accuracy = tf.reduce_mean(tf.cast(pred == label, tf.float32))
        uniques_pred = tf.unique_with_counts(pred)
        uniques_label = tf.unique_with_counts(label)

        with tf.Session() as s:
            i = 0
            tf.initialize_all_variables().eval()
            while True:
                i += 1
                _, train_accuracy, train_uniques_pred, train_uniques_label = \
                                    s.run([opt, accuracy, uniques_pred, uniques_label], feed_dict={})
                print("--")
                print("train_accuracy:      {}".format(train_accuracy))
                print("train_uniques_pred:  {}".format(train_uniques_pred))
                print("train_uniques_label: {}".format(train_uniques_label))
                print("--")

                if i % 10 == 0 and i != 0:
                    valid_accuracy, valid_uniques_pred, valid_uniques_label = \
                                    s.run([accuracy, uniques_pred, uniques_label], feed_dict={})
                    print("--")
                    print("valid_accuracy:      {}".format(valid_accuracy))
                    print("valid_uniques_pred:  {}".format(valid_uniques_pred))
                    print("valid_uniques_label: {}".format(valid_uniques_label))
                    print("--")

    elif args.net_type == "keras_dense":
        os.environ["KERAS_BACKEND"] = "tensorflow"

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(list(train_data[0].shape[1:])), ))

        model.add(Flatten())
        model.add(Dense(1028, init="normal", activation="softmax"))
        model.add(Dense(1028, init="normal", activation="softmax"))
        model.add(Dense(2, init="normal", activation="softmax"))

        from keras import backend as K

        def values_k(k, is_pred=True):
            def values(y_true, y_pred):
                work_on = y_pred if is_pred else y_true
                pre_gather = K.sum(K.argmax(work_on, axis=1))

                if k >= 0:
                    return K.gather(pre_gather, k)
                else:
                    return K.shape(pre_gather)[0]
                    #return K.gather(K.shape(pre_gather), 0)

            return values

        model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy",
                                            values_k(-1, True),
                                            #values_k(1, True), values_k(0, False), values_k(1, False)
                                            ])

        model.fit(*train_data, batch_size=args.minibatch_size,
                  validation_data=valid_data, nb_epoch=10000, shuffle="batch")

    elif args.net_type == "keras_conv":

        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(list(train_data[0].shape[1:])), input_dtype=np.float32))
        model.add(Conv2D(32, 3, 3, init="he_normal", activation="tanh",),)
        model.add(normalization.BatchNormalization())

        model.add(Conv2D(32, 3, 3, init="he_normal", activation="tanh", ))
        model.add(normalization.BatchNormalization())


        model.add(Flatten())
        model.add(Dense(2, init="he_normal", activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

        model.fit(*train_data, batch_size=args.minibatch_size,
                  validation_data=train_data, nb_epoch=10000, shuffle="batch")

    elif args.net_type == "tflearn_resnet":
        # https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py

        assert(len(x_shape) == 4)
        assert(all([len(prepared_x[i].shape) == 4 for i in range(2)]))

        net = tflearn.input_data([None] + list(x_shape[1:]))
        shape_width_test = len(net.get_shape().as_list())
        assert shape_width_test == 4, shape_width_test

        #net = tflearn.conv_2d(net, 32, 3, weight_decay=0.0001, activation="tanh")
        #net = tflearn.residual_block(net, 32, 3, weight_decay=0.0001, activation="tanh")

        #net = tflearn.batch_normalization(net)
        # Regression

        net = tflearn.fully_connected(net, 1028, activation='relu', bias=False)
        net = tflearn.fully_connected(net, 1028, activation='relu', bias=False)
        net = tflearn.batch_normalization(net, gamma=0)
        net = tflearn.fully_connected(net, 2, activation='softmax', bias=False)

        net = tflearn.regression(net, optimizer=tflearn.Adam(), loss='categorical_crossentropy',
                                metric=tflearn_Prediction_Counts(tflearn.metrics.Accuracy())
                                 )

        # Training
        model = tflearn.DNN(net, max_checkpoints=2, tensorboard_verbose=1, clip_gradients=0,
                            checkpoint_path=(args.model_save_path if
                                             args.model_save_path else
                                             'model_tflearn_resnet'))

        if args.load_model:
            model.load(args.load_model)
            print("Model metrics:\n{}\n".format(model.evaluate(valid_data[0],
                                                               valid_data[1], 32)))


        model.fit(*train_data, n_epoch=500000, shuffle=True,
                  validation_set=valid_data,
                  show_metric=True,
                  batch_size=1028)

    elif args.net_type == "tflearn_bn_vgg":
        # Our own vgg remix with bn
        print(x_shape)
        net = tflearn.input_data(x_shape[1:])  # 224x224x20
        assert np.all(net.get_shape().as_list()[1:] == [224, 224, 5]), net.get_shape().as_list()

        net = tflearn.batch_normalization(net)
        net = tflearn.conv_2d(net, 32, 3, activation='relu')
        net = tflearn.batch_normalization(net)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.conv_2d(net, 32, 3, activation='relu')
        net = tflearn.batch_normalization(net)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.conv_2d(net, 32, 3, activation='relu')
        net = tflearn.batch_normalization(net)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.max_pool_2d(net, 2, strides=2)  # 112x112x32
        assert np.all(net.get_shape().as_list()[1:] == [112, 112, 64]), net.get_shape().as_list()

        net = tflearn.batch_normalization(net)
        net = tflearn.conv_2d(net, 64, 3, activation='relu')
        net = tflearn.batch_normalization(net)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.conv_2d(net, 64, 3, activation='relu')
        net = tflearn.batch_normalization(net)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.max_pool_2d(net, 2, strides=2)  # 56x56x64
        assert np.all(net.get_shape().as_list()[1:] == [56, 56, 64]), net.get_shape().as_list()

        net = tflearn.batch_normalization(net)
        net = tflearn.conv_2d(net, 128, 3, activation='relu')
        net = tflearn.batch_normalization(net)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.max_pool_2d(net, 2, strides=2)  # 28x28x128
        assert np.all(net.get_shape().as_list()[1:] == [56, 56, 64]), net.get_shape().as_list()

        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(net, 512, activation='relu')
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(net, 2, activation='softmax')

        net = tflearn.regression(net, optimizer='rmsprop',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001)

        # Training
        model = tflearn.DNN(net, checkpoint_path='model_vgg', max_checkpoints=1, tensorboard_verbose=1)

        if args.load_model:
            model.load(args.load_model)

        model.fit(train_data[0], train_data[1], n_epoch=500, shuffle=True, validation_set=valid_data,
                  show_metric=True, batch_size=args.minibatch_size)

    elif args.net_type == "tflearn_vgg":
        # Building 'VGG Network'
        print(x_shape)
        net = tflearn.input_data(x_shape[1:])  # 224x224x20
        assert np.all(net.get_shape().as_list()[1:] == [224, 224, 5]), net.get_shape().as_list()

        net = tflearn.conv_2d(net, 64, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2, strides=2)  # 112x112x64
        assert np.all(net.get_shape().as_list()[1:] == [112, 112, 64]), net.get_shape().as_list()

        net = tflearn.conv_2d(net, 128, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2, strides=2)  # 56x56x128
        assert np.all(net.get_shape().as_list()[1:] == [56, 56, 128]), net.get_shape().as_list()

        net = tflearn.conv_2d(net, 256, 3, activation='relu')
        net = tflearn.conv_2d(net, 256, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2, strides=2)
        assert np.all(net.get_shape().as_list()[1:] == [28, 28, 256]), net.get_shape().as_list()

        net = tflearn.conv_2d(net, 512, 3, activation='relu')
        net = tflearn.conv_2d(net, 512, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2, strides=2)
        assert np.all(net.get_shape().as_list()[1:] == [14, 14, 512]), net.get_shape().as_list()

        net = tflearn.conv_2d(net, 512, 3, activation='relu')
        net = tflearn.conv_2d(net, 512, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2, strides=2)
        assert np.all(net.get_shape().as_list()[1:] == [7, 7, 512]), net.get_shape().as_list()

        net = tflearn.fully_connected(net, 4096, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, 4096, activation='relu')
        net = tflearn.dropout(net, 0.5)
        net = tflearn.fully_connected(net, 2, activation='softmax')

        net = tflearn.regression(net, optimizer='rmsprop', loss='categorical_crossentropy',
                                 learning_rate=0.001)

        # Training
        model = tflearn.DNN(net, checkpoint_path='model_vgg', max_checkpoints=1, tensorboard_verbose=1)

        if args.load_model:
            model.load(args.load_model)

        model.fit(train_data[0], train_data[1], n_epoch=500, shuffle=True, validation_set=valid_data,
                  show_metric=True, batch_size=args.minibatch_size)

    elif args.net_type == "tflearn_lstm": # should really be tflearn_lstm_resnet, but is too long for the moment
        assert False
        maxlen = 128
        seq_X = None
        seq_Y = None
        sample_bounds = None
        char_idx = None

        for bounds in sample_bounds:
            seq_X.append(training_prepared_x[bounds[0], bounds[1]])
            seq_Y.append(training_y[bounds[0], bounds[1]])

        print(np.argmax(seq_X[100], axis=0))
        print(np.argmax(seq_Y[100]))
        print("net")
        net = tflearn.input_data(X.shape[1:])
        print(net.get_shape().as_list())
        print("lstm")
        net = tflearn.lstm(net, 256, return_seq=True, activation='relu')
        net = tflearn.lstm(net, 256, activation='relu')
        print(net.get_shape().as_list())
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(net, len(char_idx), activation='softmax')
        net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',
                                 learning_rate=0.0001)

        model = tflearn.SequenceGenerator(net, dictionary=char_idx,
                                          seq_maxlen=maxlen,
                                          clip_gradients=5.0,
                                          )

        for i in range(50):
            seed = tflearn.data_utils.random_sequence_from_string(string, maxlen)
            model.fit(X, Y, validation_set=0, batch_size=args.minibatch_size,
                      n_epoch=1, run_id='fizzbuzz', snapshot_epoch=False, shuffle=False)
            test_size = 300
            print("-- TESTING...")
            print("-- Test with temperature of 10.0 --")
            check(model.generate(test_size, temperature=10.0, seq_seed=seed)[maxlen:])
            print("-- Test with temperature of 1.0 --")
            check(model.generate(test_size, temperature=1.0, seq_seed=seed)[maxlen:])
            print("-- Test with temperature of 0.1 --")
            check(model.generate(test_size, temperature=0.1, seq_seed=seed)[maxlen:])
            print("-- Test with temperature of 0.01 --")
            check(model.generate(test_size, temperature=0.001, seq_seed=seed)[maxlen:])

    elif args.net_type == "vgg":
        print("vgg - not really vgg")
        print("x_shape: {}".format(x_shape))
        random_extension_string = str(random.randint(0, 1000000000000))
        summary_path = os.path.join(base_path, "saves", "tf_summaries", "vgg_" + image_save_name + "_" +
                                    random_extension_string)

        model = nn_models.VGG(x_shape, y_shape_1, dropout_keep_prob=args.dropout_keep_prob,
                              summary_writing_path=summary_path, expected_minibatch_size=args.minibatch_size)

        model.fit(train_data[0], train_data[1], valid_data[0], valid_data[1], n_epochs=10000000,
                  minibatch_size=args.minibatch_size, learning_rate=args.learning_rate, test_qty=args.test_qty)

    elif args.net_type == "cnn":
        print("cnn")

        random_extension_string = str(random.randint(0, 1000000000000))
        summary_path = os.path.join(base_path, "saves", "tf_summaries", "cnn_" + image_name + "_" +
                                    random_extension_string)

        model = nn_models.CNN(x_shape, y_shape_1, dropout_keep_prob=args.dropout_keep_prob,
                              summary_writing_path=summary_path, expected_minibatch_size=args.minibatch_size)

        model.fit(train_data[0], train_data[1], valid_data[0], train_data[1], n_epochs=10000000,
                  minibatch_size=args.minibatch_size, learning_rate=args.learning_rate, test_qty=args.test_qty)

    elif args.net_type == "resnet":
        print("resnet")
        summary_path = gen_summ_name("resnet")

        model = nn_models.ResNet(x_shape, y_shape_1,
                                 dropout_keep_prob=args.dropout_keep_prob,
                                 summary_writing_path=summary_path,
                                 expected_minibatch_size=args.minibatch_size)

        model.fit(train_data[0], train_data[1], valid_data[0], valid_data[1], n_epochs=10000000,
                  minibatch_size=args.minibatch_size, learning_rate=args.learning_rate, test_qty=args.test_qty)


    else:
        raise RuntimeError("Received unexpected value '{}' for option --net_type".format(args.net_type))