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
import NN_models

# External
import tflearn
import tensorflow as tf
import numpy as np
import scipy.interpolate
import joblib
from mne.channels.layout import pick_types, _auto_topomap_coords

import h5py
base_path = os.path.dirname(os.path.realpath(__file__))


def _aux_interp(sample_idx, x, sample_set_idx, no_samples, no_fft_bands, sensor_positions, grid, method, picks, show, interp_x):
    if sample_idx % 5 == 0:
        sys.stdout.write("\r\t- cv set: '{}', sample: {} of {} ({:4.2f} %))".format(
            sample_set_idx, sample_idx, no_samples, 100 * sample_idx / no_samples))
        sys.stdout.flush()

    buffer_shape = interp_x[sample_set_idx][sample_idx].shape
    buffer_ = np.empty(buffer_shape)
    for fft_channel_idx in xrange(no_fft_bands):
        point_values = x[sample_set_idx][sample_idx, fft_channel_idx, picks]
        buffer_[:, :, fft_channel_idx] = scipy.interpolate.griddata(sensor_positions, point_values, grid, method, 0)

    # It is said in h5py that h5py doesn't take numpy's fancy indexing very well. Even if most of it is supported,
    # it is still pretty slow.
    # As such, we have found that creating a numpy buffer and writing it in a continuous manner to the hdf5 file
    # is really much much faster.
    # This _really_ improves performance.
    interp_x[sample_set_idx][sample_idx] = buffer_


def make_interpolated_data(x, res, method, info, sensor_type, show, hdf5_saver_loader):
    # Take any valid file's position information, as all raws [are supposed to] have the same positions

    if sensor_type == "grad":
        picks = list(range(x[0].shape[2]))
        fake_picks = pick_types(info, meg="grad")

        no_chs = len(picks)
        sensor_positions = _auto_topomap_coords(info, fake_picks, True)[:no_chs * 2:2]
        assert sensor_positions.shape[0] == no_chs, (sensor_positions.shape, no_chs)
        assert x[0].shape[2] == no_chs, (x[0].shape[2], no_chs)

        # Make sure all positions are unique
        no_positions = sensor_positions.shape[0]
        uniques = np.vstack({tuple(row) for row in sensor_positions})
        assert no_positions == uniques.shape[0]

    else:
        picks = pick_types(info, meg=sensor_type)
        sensor_positions = _auto_topomap_coords(info, picks)

    assert len(sensor_positions.shape) == 2 and sensor_positions.shape[1] == 2, sensor_positions.shape[1]
    min_x = np.floor(np.min(sensor_positions[:, 0]))
    max_x = np.ceil(np.max(sensor_positions[:, 0]))
    min_y = np.floor(np.min(sensor_positions[:, 1]))
    max_y = np.ceil(np.max(sensor_positions[:, 1]))

    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, res[0],), np.linspace(min_y, max_y, res[1]))
    grid = (grid_x, grid_y)

    h5_f = h5py.File(hdf5_saver_loader.save_path, "w", libver="latest")
    interp_x = [None, None]
    acceptable = {"grad", "mag"}

    assert sensor_type in acceptable, "sensor_type must be grad or mag, True (both) is not currently supported. " \
                                      "Got {}.".format(sensor_type)

    parr = True # This is only put to False for debugging purposes

    with joblib.Parallel(n_jobs=32, backend="threading") as pool:
        for cv_set in range(2):
            no_samples = x[cv_set].shape[utils.X_Dims.samples_and_times.value]
            no_fft_bands = x[cv_set].shape[utils.X_Dims.fft_ch.value]

            shape = [no_samples, res[0], res[1], no_fft_bands]
            interp_x[cv_set] = h5_f.create_dataset(str(cv_set), shape, np.float32)

            constant_args = dict(x=x, sample_set_idx=cv_set, no_samples=no_samples, no_fft_bands=no_fft_bands,
                                 sensor_positions=sensor_positions, grid=grid, method=method, picks=picks,
                                 show=show, interp_x=interp_x)

            if parr:
                pool(joblib.delayed(_aux_interp)(sample_idx=sample_idx, **constant_args)
                     for sample_idx in xrange(no_samples))

            else:
                start = time.time()
                for sample_idx in xrange(no_samples):
                    _aux_interp(sample_idx=sample_idx, **constant_args)

                    if sample_idx % 1000 == 0 and sample_idx != 0:
                        sys.stderr.write("\ntook {} s for 1000\n".format(time.time() - start))
                        sys.stderr.flush()
                        start = time.time()

            assert np.all(np.isfinite(interp_x[cv_set]))

            print(interp_x[cv_set].shape)
    h5_f.close()


def _make_image_save_name(res, sensor_type, nfft, fmax, tincr, use_established_bands):
    # we right them all, on purpose, instead of using *args, to make sure everything is in its place
    args = ["_".join([str(d) for d in res]), sensor_type, nfft, fmax, tincr, use_established_bands]
    return "_".join([str(x) for x in args])


class prediction_counts(tflearn.metrics.Metric):
    """ Prints the number of each kind of prediction """
    def __init__(self, name=None):
        super(Counts, self).__init__(name)

    def build(self, predictions, targets, inputs=None):
        """ Prints the number of each kind of prediction """
        self.built = True
        pshape = predictions.get_shape()

        if len(pshape) == 1 or (len(pshape) == 2 and int(pshape[1]) == 1):
            self.name = self.name or "binary_counts"
            self.tensor = tf.unique_with_counts(predictions)
        else:
            self.name = self.name or "categorical_counts"
            self.tensor = self.tensor = tf.unique_with_counts(tf.argmax(predictions, dimension=1))

        self.tensor.m_name = self.name


def spatial_classification(args):
    saves_loc = os.path.join(base_path, "saves/interp_image_saves")
    if not os.path.exists(saves_loc):
        os.mkdir(saves_loc)

    image_name = _make_image_save_name(args.res, args.sensor_type, args.nfft, args.tincr, args.fmax, args.established_bands)
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
        make_interpolated_data(x=args.x, res=args.res, method="cubic", info=args.info,
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

    training_prepared_x = prepared_x[0]
    training_y = args.y[0]

    validation_prepared_x = prepared_x[1]
    validation_y = args.y[1]

    x_shape = training_prepared_x.shape
    y_shape_1 = training_y.shape[1]

    if args.net_type == "tflearn_resnet":
        # https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py


        assert(len(x_shape) == 4)
        assert(all([len(prepared_x[i].shape) == 4 for i in range(2)]))

        net = tflearn.input_data(x_shape[1:])
        print("Shape is: {}".format(net.get_shape().as_list()))
        shape_width_test = len(net.get_shape().as_list())
        assert shape_width_test == 4, "expected 4, got {}".format(shape_width_test)

        net = tflearn.conv_2d(net, 32, 3, weight_decay=0.0001)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.batch_normalization(net)

        net = tflearn.residual_block(net, 2, 32, bias=False)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.max_pool_2d(net, 3, strides=2)
        assert net.get_shape().as_list()[1:] == [112, 112, 32], net.get_shape().as_list()

        net = tflearn.residual_block(net, 2, 64, bias=False)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.max_pool_2d(net, 3, strides=2)
        assert net.get_shape().as_list()[1:] == [56, 56, 64], net.get_shape().as_list()

        net = tflearn.residual_block(net, 1, 128, bias=False)
        net = tflearn.dropout(net, args.dropout_keep_prob)
        net = tflearn.max_pool_2d(net, 3, strides=2)
        assert net.get_shape().as_list()[1:] == [28, 28, 128], net.get_shape().as_list()

        net = tflearn.batch_normalization(net)
        net = tflearn.activation(net, 'relu')
        net = tflearn.global_avg_pool(net)

        # Regression
        net = tflearn.fully_connected(net, 2, activation='softmax')
        mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
        net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')

        # Training
        model = tflearn.DNN(net, max_checkpoints=2, tensorboard_verbose=1, clip_gradients=0,
                            checkpoint_path=(args.model_save_path if args.model_save_path else 'model_tflearn_resnet')
                            )

        if args.load_model:
            model.load(args.load_model)
            print("Model metrics:\n{}\n".format(model.evaluate(validation_prepared_x, validation_y, 32)))

        model.fit(training_prepared_x, training_y,  n_epoch=500, shuffle=True,
                  validation_set=(validation_prepared_x, validation_y),
                  show_metric=[True, tflearn.metrics.accuracy, prediction_counts], batch_size=32,)
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

        model.fit(training_prepared_x, training_y, n_epoch=500, shuffle=True,
                  validation_set=(validation_prepared_x, validation_y),
                  show_metric=True, batch_size=32, )

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

        net = tflearn.regression(net, optimizer='rmsprop',
                                 loss='categorical_crossentropy',
                                 learning_rate=0.001)

        # Training
        model = tflearn.DNN(net, checkpoint_path='model_vgg', max_checkpoints=1, tensorboard_verbose=1)

        if args.load_model:
            model.load(args.load_model)

        model.fit(training_prepared_x, training_y, n_epoch=500, shuffle=True,
                  validation_set=(validation_prepared_x, validation_y),
                  show_metric=True, batch_size=32, )

    elif args.net_type == "tflearn_lstm": # should really be tflearn_lstm_resnet, but is too long for the moment

        maxlen = 128
        dataset_size = 500000
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
            model.fit(X, Y, validation_set=0, batch_size=32,
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
        summary_path = os.path.join(base_path, "saves", "tf_summaries", "vgg_" + image_save_name + "_" +
                                    str(random.randint(0, 1000000000000)))
        model = NN_models.VGG(x_shape, y_shape_1, dropout_keep_prob=args.dropout_keep_prob,
                              summary_writing_path=summary_path, expected_minibatch_size=args.minibatch_size)
        if not args.dry_run:
            model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                      n_epochs=10000000, minibatch_size=args.minibatch_size, learning_rate=args.learning_rate,
                      test_qty=args.test_qty)
        else:
            print(">>>>> NO FITTING, WAS A DRY RUN")

    elif args.net_type == "cnn":
        print("cnn")

        summary_path = os.path.join(base_path, "saves", "tf_summaries", "cnn_" + image_name + "_" +
                                    str(random.randint(0, 1000000000000)))
        model = NN_models.CNN(x_shape, y_shape_1, depth=args.depth, dropout_keep_prob=args.dropout_keep_prob,
                              filter_scale_factor=args.filter_scale_factor, summary_writing_path=summary_path,
                              expected_minibatch_size=args.minibatch_size)
        if not args.dry_run:

            model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y, n_epochs=10000000,
                      minibatch_size=args.minibatch_size, learning_rate=args.learning_rate, test_qty=args.test_qty)
        else:
            print(">>>>> NO FITTING, WAS A DRY RUN")
    elif args.net_type == "resnet":
        print("resnet")
        summary_path = os.path.join(base_path, "saves", "tf_summaries", "resnet_" + image_save_name + "_" +
                                    str(random.randint(0, 1000000000000)))
        model = NN_models.ResNet(x_shape, y_shape_1, dropout_keep_prob=args.dropout_keep_prob,
                                 summary_writing_path=summary_path,
                                 expected_minibatch_size=args.minibatch_size)
        if not args.dry_run:
            model.fit(training_prepared_x, training_y, validation_prepared_x, validation_y,
                      n_epochs=10000000, minibatch_size=args.minibatch_size, learning_rate=args.learning_rate, test_qty=args.test_qty)
        else:
            print(">>>>> NO FITTING, WAS A DRY RUN")


    else:
        raise RuntimeError("Received unexpected value '{}' for option --net_type".format(args.net_type))