#! /usr/bin/env python

# Compatibility imports
from __future__ import with_statement, print_function, division
from six import iteritems
from six.moves import zip as izip

# Stdlib imports
import sys, os, argparse, time, logging, enum, json, subprocess as sp
from collections import defaultdict as dd, Counter


# Adding the module root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import *

# scipy/numpy/matplotlib/tf
import numpy as np
import scipy
import matplotlib as mpl
import tensorflow as tf

# MNE imports
import mne, mne.time_frequency
from mne.decoding import CSP
from mne.channels.layout import _auto_topomap_coords

# Sklearn imports
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import logistic
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Varia
import tflearn
import h5py

"""
mne's logger is way to talkative, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) work.
"""
logger = logging.getLogger('mne')
logger.disabled = True


class X_Dims(enum.Enum):
    samples_and_times = 0
    fft_ch = 1
    sensors = 2
    size = 3


def maybe_prep_psds(args):

    # Display the args
    print("\nArgs:" )
    for k, v in iteritems(vars(args)):
        print("--{k}:".format(k=k).ljust(20, " ") + "{v}".format(v=v))
    print("")

    # Warn if some values are weird
    if args.min_procs != 1 or args.max_procs != 1:
        print("Warning: --min_procs and --max_procs should probably both be 1, " \
              "or left alone, as benchmarks say more procs decrease performance.")

    if args.reps != 1:
        print("Warning: --rep should be 1 or left alone, unless you want to test the " \
              "performance of the psd function, which there is no real reason to do right now."\
              "Value: {}".format(args.reps))

    if args.glob_tmin != 0:
        print("Warning: --glob_tmin is not equal to zero, this is weid. Value : {}".format(args.glob_tmin))

    # We assign the values we obtained
    MIN_PROCS    = args.min_procs
    MAX_PROCS    = args.max_procs
    NFFT         = args.nfft
    GLOB_TMIN    = args.glob_tmin
    GLOB_TMAX    = args.glob_tmax
    GLOB_TINCR   = args.glob_tincr
    NOVERLAP     = args.noverlap
    DATA_PATH    = args.data_path

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################
    print("# FEATURE PREPARATION")

    checked_min_procs = MIN_PROCS if MIN_PROCS else 1 # 1 if MIN_PROCS is 0 or None

    X = [None, None, None] # Generated PSDs
    Y = [[], [], []] # Generated labels

    BASE_PATH = os.path.dirname(__file__)
    json_path = os.path.join(BASE_PATH, "fif_split.json")
    print("#2: {}".format(json_path))
    with open(json_path, "r") as json_f:
        fif_split = json.load(json_f) # dict with colors

    split_idx_to_name = {"training":  0,
                         "valid":     1,
                         "test":      2,
                         }

    for name, raw, label in data_gen(DATA_PATH):
        split_idx = split_idx_to_name[fif_split[name]]

        outer_time_bound = raw.n_times / 1000.
        for procs_to_use in range(checked_min_procs, MAX_PROCS + 1):
            for psd_band_t_start in range(GLOB_TMIN, GLOB_TMAX + 1, GLOB_TINCR):
                if outer_time_bound < psd_band_t_start:
                    # reg_print("{} < {} ; rejected".format(outer_time_bound, psd_band_t_start))
                    break

                # So, the point here is that we don't want to crash if the raw is malformed.
                # However, just catching all ValueError's is really too permissive; we need to be more precise here.
                num_res, freqs = mne.time_frequency.psd_welch(
                                           n_jobs=procs_to_use,
                                           inst=raw,
                                           picks=mne.pick_types(raw.info, meg=True),
                                           n_fft=NFFT,
                                           n_overlap=NOVERLAP,
                                           tmin=psd_band_t_start,
                                           tmax=psd_band_t_start + GLOB_TINCR,
                                           verbose="INFO"
                                           )

                num_res = 10.0 * np.log10(num_res)

                if X[split_idx] is None:
                    X[split_idx] = num_res

                else:
                    if num_res.shape[X_Dims.fft_ch.value] == X[split_idx].shape[X_Dims.fft_ch.value]: # All samples need the same qty of fft channels
                        X[split_idx] = np.dstack([X[split_idx], num_res])
                    else:
                        print("num_res of bad shape '{}' rejected, should be {}".format(num_res.shape, X[split_idx].shape[:2]))
                        continue

                Y[split_idx].append(label)

    assert len(X) == 3
    assert len(Y) == 3

    for i in xrange(3):
        assert type(X[i]) == np.ndarray

        # We convert the PSD list of ndarrays to a single multidimensional ndarray
        X[i] = X[i].astype(np.float32)

        # We do the same with the labels
        Y[i] = np.asarray(Y[i], np.float32)

        # Transpose for convenience
        X[i] = X[i].T

        # Center and normalise
        X[i] = (X[i] - np.mean(X[i]))
        X[i] = X[i] / np.std(X[i])


        assert len(X[i].shape) == X_Dims.size.value # meh
        assert X[i].shape[X_Dims.samples_and_times.value] == Y[i].shape[0], X[i].shape[X_Dims.samples_and_times.value]  # no_samples
        assert X[i].shape[X_Dims.sensors.value] == 306, X[i].shape[X_Dims.sensors.value]  # sensor no

        print("X[{}].shape = {}".format(i, X[i].shape))
        print("Y[{}].shape = {}".format(i, Y[i].shape))

    # Take any valid file's position information, as all raws [are supposed to] have the same positions
    info = next(data_gen(DATA_PATH))[1].info

    return X, Y, info

def parse_args(argv):
    # Doing the arg parsing here is horrible. This is for this iteration only.
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################
    # DEFAULTS FOR THE ARGUMENTS
    d_REP = 1  # Number of repetitions, for timeit
    d_GLOB_TINCR = 10  # Length of the time clips on which the PSDs are calculated
    d_NOVERLAP = 0  # Overlap between the time clips on which the PSDs are calculated
    d_NFFT = 2048  # Quantity of fft bands

    # Defaults that are used
    ## welch is much faster, like 50x faster. There is more leakage than multitaper, but the performance hit
    ## is way too big for the moment for it to be justified.
    d_F = "welch"
    ## procs_to_use: 1 is the fastest for our data size for both our pc and the cluster... ##
    d_MIN_PROCS = 1
    d_MAX_PROCS = 1
    # we're not dumping any part of the features
    d_GLOB_TMIN = 0
    d_GLOB_TMAX = 1000

    # ARGUMENT PARSING
    p = argparse.ArgumentParser(argv)
    p.add_argument("--nfft", type=int, default=d_NFFT)
    p.add_argument("--glob_tincr", type=int, default=d_GLOB_TINCR)
    p.add_argument("--noverlap", type=int, default=d_NOVERLAP)
    p.add_argument("-o", "--data_path", type=str)

    # These don'psd_time_band_start need to be explored anymore (for the moment at least).
    p.add_argument("-r", "--reps", type=int, default=d_REP)
    p.add_argument("--min_procs", type=int, default=d_MIN_PROCS)
    p.add_argument("--max_procs", type=int, default=d_MAX_PROCS)
    p.add_argument("-f", "--funcs", type=str, default=d_F)
    p.add_argument("--glob_tmin", type=int, default=d_GLOB_TMIN)
    p.add_argument("--glob_tmax", type=int, default=d_GLOB_TMAX)

    return p.parse_args(argv[1:])

def make_samples_linear(X, Y):
    linear_X = X.reshape(X.shape[X_Dims.samples_and_times.value],  X.shape[X_Dims.fft_ch.value] * X.shape[X_Dims.sensors.value])
    return linear_X, Y


def to_one_hot(input, max_classes):
    no_samples = input.shape[0]
    output = np.zeros((input.shape[0], max_classes), np.float32)
    output[np.arange(no_samples), input.astype(np.int32)] = 1
    return output


def from_one_hot(values):
    return np.argmax(values, axis=1)


class AbstractTensorflowLinearClassifier(object):
    def fit(self, linear_x, linear_y, n_epoch, minibatch_size=64):
        for i in xrange(3):
            std_x = np.std(linear_x[i])
            assert std_x > 1E-3, std_x
            std_y = np.std(linear_y[i])
            assert std_y > 1E-3, std_y

        assert type(minibatch_size) == int, "minibatch_size should be an integer. it's currently an {}".format(
            type(n_epoch))
        assert minibatch_size > 0, "n_epoch should be larger than zero. its value is currently '{}'".format(n_epoch)
        is_power_of_2 = minibatch_size & (minibatch_size - 1) == 0
        assert is_power_of_2, "minibatch_size should be an integer power of 2. It's log2 is currently '{}', with " \
                              "it's value being '{}'.".format(
            np.log2(minibatch_size), minibatch_size)

        training_x = linear_x[0]
        training_y = linear_y[0]

        output = h5py.File(os.path.dirname(__file__) + "/scores/" + "{}_{}.h5".format(self._type, time.time()), "a",
                           libver='latest',
                           compression=None)

        set_names = ["scores_training", "scores_valid"]
        output.create_dataset(set_names[0], data=np.nan * np.ones((n_epoch,), np.float32))
        output.create_dataset(set_names[1], data=np.nan * np.ones((n_epoch,), np.float32))

        with tf.Session() as s:
            s.run([tf.initialize_all_variables()])
            for i in xrange(n_epoch):


                for j in xrange(0, training_x.shape[0] // minibatch_size + 1):
                    idx_from = j * minibatch_size
                    idx_to = min((j + 1) * minibatch_size, training_x.shape[0] - 1)
                    diff = idx_to - idx_from

                    if diff == 0:
                        print("diff == 0, skipping")
                        break

                    loss, opt = s.run([self.loss, self.opt],
                                      feed_dict={
                                          self.x_ph: training_x[idx_from:idx_to, :],
                                          self.y_ph: training_y[idx_from:idx_to, :]
                                      }
                                      )


                # Save both the training and validation score to hdf5
                if i % 100 == 0 and i != 0:
                    print("EPOCH {}".format(i))
                    sys.stdout.write("{_type}::{epoch}: {loss}\n".format(_type=self._type, epoch=i, loss=loss))

                    for set_id in xrange(2):
                        preds = s.run(self.classif, feed_dict={self.x_ph: linear_x[set_id]})
                        decision = np.argmax(preds, axis=1)
                        label = np.argmax(linear_y[set_id], axis=1)
                        score = np.mean(label == decision)
                        set_name = set_names[set_id]
                        output[set_name][i] = score
                        sys.stdout.write("{_type}::{epoch}::{set_name}: {score}\n".format(_type=self._type,
                                                set_name=set_name, epoch=i, score=score))

                        if i % 1000 == 0 and i != 0:
                            print("FLUSHING")
                            output.flush()
                            sys.stdout.flush()


class LogReg(AbstractTensorflowLinearClassifier):
    def __init__(self, input_ph_shape, output_ph_shape):
        self._type = "LogReg"

        self.x_ph = tf.placeholder(dtype=tf.float32, shape=input_ph_shape)
        self.y_ph = tf.placeholder(dtype=tf.float32, shape=output_ph_shape)

        w0_s = (input_ph_shape[1], output_ph_shape[1])
        b0_s = (output_ph_shape[1],)  #

        self.w0 = tf.Variable(initial_value=tf.truncated_normal(w0_s), dtype=tf.float32)
        self.b0 = tf.Variable(initial_value=tf.truncated_normal(b0_s), dtype=tf.float32)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            tf.matmul(self.x_ph, self.w0) + self.b0, self.y_ph)) + 0.01 * tf.nn.l2_loss(self.w0)

        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.classif = tf.nn.softmax(tf.matmul(self.x_ph, self.w0) + self.b0)


class FFNN(AbstractTensorflowLinearClassifier):
    def __init__(self, input_ph_shape, output_ph_shape):
        self._type = "FFNN"

        self.x_ph = tf.placeholder(dtype=tf.float32, shape=input_ph_shape)
        self.y_ph = tf.placeholder(dtype=tf.float32, shape=output_ph_shape)
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=tuple())

        w0_s = (input_ph_shape[1], 1024)
        b0_s = (w0_s[1],)

        self.w0 = tf.Variable(initial_value=tf.truncated_normal(w0_s), dtype=tf.float32)
        self.b0 = tf.Variable(initial_value=tf.truncated_normal(b0_s), dtype=tf.float32)

        w1_s = (w0_s[1], 1024)
        b1_s = (w1_s[1],)

        self.w1 = tf.Variable(initial_value=tf.truncated_normal(w1_s), dtype=tf.float32)
        self.b1 = tf.Variable(initial_value=tf.truncated_normal(b1_s), dtype=tf.float32)

        w2_s = (w1_s[1], output_ph_shape[1])
        b2_s = (output_ph_shape[1],)

        self.w2 = tf.Variable(initial_value=tf.truncated_normal(w2_s), dtype=tf.float32)
        self.b2 = tf.Variable(initial_value=tf.truncated_normal(b2_s), dtype=tf.float32)

        self.l0 = tf.nn.relu(tf.matmul(self.x_ph, self.w0) + self.b0)
        self.d0 = tf.nn.dropout(self.l0, self.dropout_keep_prob)
        self.l1 = tf.nn.relu(tf.matmul(self.d0, self.w1) + self.b1)
        self.d1 = tf.nn.dropout(self.l1, self.dropout_keep_prob)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          tf.matmul(self.d1, self.w2) + self.b2, self.y_ph))

        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)

        self.classif = tf.nn.softmax(tf.matmul(self.x_ph, self.w0) + self.b0)


def linear_classification(linear_x, linear_y):
    header = ("*********************************************************\n"
              "**** Classification Code :                               \n"
              "*********************************************************")
    print(header)
    print(header)

    assert len(linear_x) == 3
    assert len(linear_y) == 3

    feature_width = linear_x[0].shape[1]

    training_x = linear_x[0]
    valid_x = linear_x[1]
    test_x = linear_x[2]

    training_y = linear_y[0]
    valid_y = linear_y[1]
    test_y = linear_y[2]

    classifiers = [
                    LogReg([None, feature_width], [None, 2]),  # The first None is the Number of minibatches. .. The second one too.
                    FFNN([None, feature_width], [None, 2]),
                    SVC(kernel="poly", degree=3),
                    SVC(kernel="poly", degree=30),
                    SVC(kernel="poly", degree=300),
                    SVC(kernel="linear"),
                    SVC(kernel="linear", C=0.1),
                    SVC(kernel="linear", C=0.01),
                    SVC(kernel="linear", C=0.001),
                    SVC(kernel="rbf"),
                    LinearSVC(),
                    logistic.LogisticRegression(),
                    RandomForestClassifier(n_estimators=10),
                    KNeighborsClassifier(),
                    ]

    one_hot_set = {tflearn.DNN, LogReg, FFNN}
    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            one_hot_y = [to_one_hot(_y, 2) for _y in linear_y]
            classifier.fit(linear_x, one_hot_y, n_epoch=300000) #, validation_set=(features_va, labels_va))

            if type(classifier) == tflearn.DNN:
                predicted_valid_y = np.argmax(classifier.predict(valid_x), axis=1)
                predicted_train_y = np.argmax(classifier.predict(training_x), axis=1)

                print("-------------------------------------")
                print("classifier: {}".format(classifier))
                print("training score: {}".format(np.mean(predicted_train_y == one_hot_y[0])))
                print("valid score: {}".format(np.mean(predicted_valid_y == one_hot_y[1])))
                print("-------------------------------------")

        else:
            labels_tr = training_y
            cl = classifier.fit(training_x, labels_tr)

            print("-------------------------------------")
            print("classifier: {}".format(classifier))
            print("valid score: {}".format(cl.score(training_x, training_y)))
            print("valid score: {}".format(cl.score(valid_x, valid_y)))
            print("-------------------------------------")


def spatial_classification(interp_X, interp_Y, train_picks, valid_picks, test_picks):
    assert False, "This code is not functional"
    # normalization
    interp_X = (interp_X - np.average(interp_X)) / np.std(interp_X)

    # Real-time data augmentation
    # img_aug = tflearn.ImageAugmentation()
    # img_aug.add_random_flip_leftright()
    # img_aug.add_random_rotation(max_angle=5)

    # img_prep = ImagePreprocessing()
    # img_prep.add_featurewise_zero_center()
    # img_prep.add_featurewise_stdnorm()

    # Convolutional network building
    network = tflearn.input_data(shape=[None, 32, 32, 3],
                         #data_preprocessing=img_prep,
                         #data_augmentation=img_aug
                                 )
    network = tflearn.conv_2d(network, 32, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.conv_2d(network, 64, 3, activation='relu')
    network = tflearn.max_pool_2d(network, 2)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 10, activation='softmax')
    network = tflearn.regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)

    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(interp_X, interp_Y, n_epoch=50, shuffle=True, validation_set=(interp_X[valid_picks], interp_Y[valid_picks]),
              show_metric=True, batch_size=96)

    classifiers = [model]
    one_hot_set = {model}

    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            features_tr = interp_X[train_picks, :]
            labels_tr = to_one_hot(interp_Y[train_picks], 2)

            features_va = interp_X[valid_picks]
            labels_va = to_one_hot(interp_Y[valid_picks], 2)

            classifier.fit(features_tr, labels_tr, n_epoch=10000, validation_set=(features_va, labels_va))

            predicted_va = np.argmax(classifier.predict(features_va), axis=1)

            print(np.mean(predicted_va == labels_va))
            print(Counter(predicted_va))
            print(Counter(interp_Y[valid_picks].tolist()))

        else:
            raise RuntimeError("Landed in a dead section")


# https://github.com/mne-tools/mne-python/blob/master/mne/viz/topomap.py
def make_interpolated_data(X, res, method, sample_info, hdf5_save_path, sensor_type="grad", show=False):
    picks = mne.pick_types(sample_info, meg=sensor_type)
    sensor_positions = _auto_topomap_coords(sample_info, picks, True)


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

    # FIRST DIM IS 1 IF SHOW IS ENABLED

    BASE_PATH = os.path.dirname(__file__)
    h5_file = h5py.File(os.path.join(BASE_PATH, "interpolated_data", "{}.h5".format(time.time())))
    interp_x = h5_file.create_dataset("interpolated", shape=(1 if show else X.shape[X_Dims.samples_and_times.value], X.shape[X_Dims.fft_ch.value], res[0], res[1]), dtype=np.float32)

    for i in xrange(X.shape[X_Dims.samples_and_times.value]):
        for j in xrange(X.shape[X_Dims.fft_ch.value]):

            psd_image = griddata(sensor_positions[picks, :], X[i, j, picks], grid, method)
            interp_x[i, j, :] = psd_image[:, :]

            if show:
                plt.imshow(psd_image, interpolation="none")
                plt.show()

    h5_file.flush()
    return interp_x


def main(argv):
    start = time.time()
    BASE_PATH = os.path.dirname(__file__)

    # If we don't have arguments, try to load last known config.
    # If we do have argumetns, save them as the last known config.
    if len(argv) <= 1:
        with open(os.path.join(BASE_PATH, "direct_args.json"), "r") as _if:
            argv = json.load(_if)
    else:
        with open(os.path.join(BASE_PATH, "direct_args.json"), "w") as _if:
            json.dump(argv, _if)

    args = parse_args(argv)



    json_split_path = os.path.join(BASE_PATH, "fif_split.json")
    print("#1: {}".format(json_split_path))

    if not os.path.exists(json_split_path):
        import generate_split
        generate_split.main([None, args.data_path, BASE_PATH])

    X, Y, sample_info = maybe_prep_psds(args) # argv being passed is temporary

    for i in xrange(3):
        print("")
        print(np.average(X[i]))
        std_dev = np.std(X[i])
        print(std_dev)
        print("")

    ###########################################################################
    # CLASSICAL MACHINE LEARNING CLASSIFICATION without locality
    ###########################################################################

    print("# CLASSICAL MACHINE LEARNING")

    linear_X = [None, None, None]
    linear_Y = [None, None, None]

    for i in xrange(3):
        linear_X[i], linear_Y[i] = make_samples_linear(X[i], Y[i])

    linear_classification(linear_X, linear_Y)

    # reg_print("# SPATIAL MACHINE LEARNING")
    # interp_X = make_interpolated_data(X, (1000, 1000), "cubic", sample_info)

    # spatial_classification(interp_X, Y, training_picks, valid_picks, test_picks)

    ###########################################################################
    # LOCALITY PRESERVING CLASSICAL MACHINE LEARNING
    ###########################################################################

    ###########################################################################
    # VGG classical style CONVOLUTIONAL NEURAL NETWORK
    ###########################################################################

    # 3x3 conv, relu, 3x3 conv, relu, 3x3 conv, relu, maxpool, 3x3 conv, relu, 3x3 conv, relu, maxpool, FC, FC
    # with batchnorm and dropout

    # TODO

    ###########################################################################
    # RESNET CONVOLUTIONAL NEURAL NETWORK
    ###########################################################################

    # TODO
    end = time.time()
    print("TOTAL TIME: {total_time} sec".format(total_time=end - start))
    print("*********************************************************\n"
          "Total: '%s'" % (end - start))

if __name__ == "__main__": main(sys.argv)
