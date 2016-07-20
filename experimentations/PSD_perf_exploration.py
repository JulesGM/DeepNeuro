from __future__ import with_statement, print_function, division

from six import iteritems
from six.moves import zip as izip

import sys, os, argparse, timeit, time, logging
from subprocess import check_output
from collections import defaultdict as dd, Counter
import scipy

import numpy as np
import mne, mne.time_frequency
from mne.decoding import CSP

import tflearn

"""
mne's logger is way to talkative, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) work.
"""
logger = logging.getLogger('mne')
logger.disabled = True

# Add module root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import *
verbose = 0

def err_print(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def reg_print(msg, verbose=None):
    print(msg)

def timed_psd_log_info(infos_dict):
    """
    Wanted to remove the logging abstractions from the computing abstractions.
    """
    if verbose > 2:
        print("---------------------------------------------------------")
        for k, v in iteritems(infos_dict):
            print(("%s:" % k).ljust(10, " ") + "'%s'" % v)
        print("---------------------------------------------------------")

def timed_psd(f, rep, **f_fkwargs):
    """
    Time the psd function, log a couple things.
    At the moment of writing this comment, the primary candidate for this is mne.time_frequency.psd_welsh, because of its
    much better performance over mne.time_frequency.psd_multitaper
    f_fkwargs are meant to be f's kwargs.
    """

    # In Python, lists are the closest thing to a pointer we have access to. Here, we use a list as a ref
    # to collect the results of the timed function, which aren't returned by the timeit function (it returns the mesured
    # execution times).
    results = []

    # In my opinion this is cleaner than a lambda LOL WTF IS THIS
    def run():
        results.append(f(**f_fkwargs))

    chrono = np.array(timeit.Timer(run).repeat(rep, 1))

    loggable_info = {}

    loggable_info["avg"]        =  np.average(chrono)
    loggable_info["min"]        = np.min(chrono)
    loggable_info["f.__name__"] = f.__name__
    loggable_info["rep"]        = rep

    loggable_info.update(f_fkwargs)
    timed_psd_log_info(loggable_info)
    psd_amps, psd_freqs = results[0]

    return chrono, psd_amps

"""
http://martinos.org/mne/stable/generated/mne.time_frequency.psd_welch.html#mne.time_frequency.psd_welch

Current plan:
    - toCONFandHDF5:
        - Make a small lib that saves a conf with an hdf5 file,
            and regenerates & saves if any of the conf has changed
    - PSDs to HDF5 :
        - toCONFandHDF5
    - INTERPOLATION:    return linear_X, linear_Ylet let let     return linear_X, linear_Ylet let let


        SIZE preoccupations:
        - toCONFandHDF5
        - Similarly, if interp conf hasn't changed, load from hdf5
        - Else, load from hdf5
        ----> THIS WILL BE OF HUGE SIZE... res**2 * nfft
        - we'll see how slow to generate this is
    - DEEPLEARNING:
        ------>
"""

import enum
class X_Dims(enum.Enum):
    samples_and_times = 0
    fft_ch = 1
    sensors = 2
    size = 3


def maybe_prep_psds(argv):
    # Doing the arg parsing here is horrible. This is for this iteration only.
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################
    # DEFAULTS FOR THE ARGUMENTS
    d_REP        = 1    # Number of repetitions, for timeit
    d_GLOB_TINCR = 10  # Length of the time clips on which the PSDs are calculated
    d_NOVERLAP   = 0    # Overlap between the time clips on which the PSDs are calculated
    d_NFFT       = 2048   # Quantity of fft bands

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
    p.add_argument(       "--nfft",        type=int,  default=d_NFFT)
    p.add_argument(       "--glob_tincr",  type=int,  default=d_GLOB_TINCR)
    p.add_argument(       "--noverlap",    type=int,  default=d_NOVERLAP)
    p.add_argument("-o",  "--data_path",   type=str)

    # These don'psd_time_band_start need to be explored anymore (for the moment at least).
    p.add_argument("-r",  "--reps",        type=int, default=d_REP)
    p.add_argument(       "--min_procs",   type=int,  default=d_MIN_PROCS)
    p.add_argument(       "--max_procs",   type=int,  default=d_MAX_PROCS)
    p.add_argument("-f",  "--funcs",       type=str,  default=d_F)
    p.add_argument(       "--glob_tmin",   type=int,  default=d_GLOB_TMIN)
    p.add_argument(       "--glob_tmax",   type=int,  default=d_GLOB_TMAX)

    args = p.parse_args()

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
    REP          = args.reps
    FUNCS        = args.funcs
    NFFT         = args.nfft
    GLOB_TMIN    = args.glob_tmin
    GLOB_TMAX    = args.glob_tmax
    GLOB_TINCR   = args.glob_tincr
    NOVERLAP     = args.noverlap
    DATA_PATH    = args.data_path

    PSD_FUNC     = mne.time_frequency.psd_welch

    # We convert the values of FUNC to actual functions
    if FUNCS == "both":
        FUNCS_list = [mne.time_frequency.psd_multitaper, mne.time_frequency.psd_welch]
    elif FUNCS == "welch" or FUNCS == "w":
        FUNCS_list = [mne.time_frequency.psd_welch]
    elif FUNCS == "mt" or FUNCS == "multitaper" or FUNCS == "m":
        FUNCS_list = [mne.time_frequency.psd_multitaper]
    else:
        raise RuntimeError("invalid funcs argument. must be one of ['both', 'w', 'm']. Got '%s'." % FUNCS)

    ###########################################################################
    # FEATURE PREPARATION
    ###########################################################################
    print("# FEATURE PREPARATION")
    # We want to time the compute time, not the data loading time.
    # The raw data is at preload = True to make this measure more relevant (less dependant on the HDD reads)

    checked_min_procs = MIN_PROCS if MIN_PROCS else 1 # 1 if MIN_PROCS is 0 or None

    X = None # Generated PSDs
    Y = [] # Generated labels

    for raw, label in data_gen(DATA_PATH):
        #raw.plot_psd(show=True)

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

                num_res = 10 * np.log10(num_res)

                #print(freqs)
                #print(num_res)
                #print("std_dev: {}".format(np.std(num_res)))
                #print("mean: {}".format(np.average(num_res)))

                if X is None:
                    X = num_res
                else:
                    if num_res.shape[X_Dims.fft_ch.value] == X.shape[X_Dims.fft_ch.value]: # All samples need the same qty of fft channels
                        X = np.dstack([X, num_res])
                        
                    else:
                        print("num_res of bad shape '{}' rejected, should be {}".format(num_res.shape, X.shape[:2]))
                        continue

                Y.append(label)

    assert X is not None, "X is None"

    # We convert the PSD list of ndarrays to a single multidimensional ndarray
    X = X.astype(np.float32)
    assert type(X) == np.ndarray and X.dtype == np.float32
    # We do the same with the labels
    Y = np.asarray(Y, np.float32)
    X = X.T

    assert len(X.shape) == X_Dims.size.value
    assert X.shape[X_Dims.samples_and_times.value] == Y.shape[0], X.shape[X_Dims.samples_and_times.value]  # no_samples
    assert X.shape[X_Dims.sensors.value] == 306, X.shape[X_Dims.sensors.value]  # sensor no

    print("X.shape = {}".format(X.shape))
    print("Y.shape = {}".format(Y.shape))


    return X, Y


def make_picks(no_samples):
    ###########################################################################
    # DATA SPLIT
    ###########################################################################
    print("# DATA SPLIT")

    # print("CURRENTLY USING CONSTANT RANDOM SEED")
    # np.random.seed(0)

    training_limit = int(.6 * no_samples)
    valid_limit = training_limit + int(.2 * no_samples)
    randomized = np.random.permutation(no_samples)
    training_picks = randomized[:training_limit]
    valid_picks = randomized[training_limit:valid_limit]
    test_picks = randomized[valid_limit:]

    return training_picks, valid_picks, test_picks


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

def stats(Y):
    y_count = Counter(Y.tolist())
    print("########################################")
    print("Count: {}".format(y_count))
    print("Ratio: {}".format(y_count[y_count.keys()[0]] / (sum(y_count.values()))))
    print("########################################")


def linear_classification(linear_X, linear_Y, train_picks, valid_picks, test_picks):
    ## Temporary, clean split in the error output

    header = ("*********************************************************\n"
              "**** Classification Code :                               \n"
              "*********************************************************")
    err_print(header)
    reg_print(header)

    linear_X = linear_X / np.std(linear_X)

    assert len(linear_X.shape) == 2

    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import logistic
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    #linear_X = np.random.rand(*linear_X.shape)

    # We currently only use sklearn classifiers and they all use the same interface, so we just build
    # the classifiers in a list and try them iteratively.
    # Eventually, we will have more precise configs for them, undoubtably
    #stats(np.array(linear_Y))

    # feedforward neural net
    logging.getLogger("tflearn")
    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.75)
    net = tflearn.input_data(shape=[None, linear_X.shape[1]])
    # net = tflearn.fully_connected(net, 100)
    # net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation="softmax")
    import tensorflow as tf
    net = tflearn.regression(net, optimizer="adam", loss="categorical_crossentropy")
    dnn = tflearn.DNN(net)




    import tensorflow as tf
    class Feedforward(object):
        def __init__(self, input_ph_shape, output_ph_shape):



            self.x_ph = tf.placeholder(dtype=tf.float32, shape=input_ph_shape)
            self.y_ph = tf.placeholder(dtype=tf.float32, shape=output_ph_shape)

            # x dim 0: samples #
            # x dim 1: feature dimensions

            # y dim 0: sample #
            # y dim 1: label dimension // one hot output

            w0_s = (input_ph_shape[1], output_ph_shape[1])      # x_[1], y_[1]
            b0_s = (output_ph_shape[1],)                      #

            self.w0 = tf.Variable(initial_value=tf.truncated_normal(w0_s), dtype=tf.float32)
            self.b0 = tf.Variable(initial_value=tf.truncated_normal(b0_s), dtype=tf.float32)


            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(self.x_ph, self.w0) + self.b0, self.y_ph) ) + tf.nn.l2_loss(self.w0)
            self.opt = tf.train.AdamOptimizer(0.00001).minimize(self.loss)

            self.classif = tf.nn.softmax(tf.matmul(self.x_ph, self.w0) + self.b0)

        def fit(self, X, Y, n_epoch, minibatch_size=64):
            std_x = np.std(X)
            assert std_x > 1E-3, std_x

            std_y = np.std(Y)
            assert std_y > 1E-3, std_y


            # tf_X = tf.convert_to_tensor(X)
            # tf_Y = tf.convert_to_tensor(Y)

            with tf.Session() as s:
                s.run([tf.initialize_all_variables()])
                for i in xrange(n_epoch):
                    print("EPOCH {}".format(i))
                    for j in xrange(0, X.shape[0] // minibatch_size + 1):
                        idx_from = j * minibatch_size
                        idx_to   = min((j + 1) * minibatch_size, X.shape[0] - 1)
                        diff = idx_to - idx_from

                        if diff == 0:
                            print("diff == 0, skipping")
                            break;

                        loss, opt = s.run([self.loss, self.opt],
                                          feed_dict={
                                                     self.x_ph: X[idx_from:idx_to, :],
                                                     self.y_ph: Y[idx_from:idx_to, :]
                                                     }
                                          )

                        print("loss: {}".format(loss))

                    preds = s.run(self.classif, feed_dict={self.x_ph: X})
                    # print("Y:")
                    # print(Y)
                    print("argmax preds:")
                    #print(np.argmax(preds, axis=1))
                    #print(preds)
                    # print("argmax Y:")
                    # print(np.argmax(Y, axis=1))

                    print("score: {}".format(np.mean(np.argmax(Y, axis=1) == np.argmax(preds, axis=1))))
                    print("mean anser: {}".format(np.mean(preds)))

    """
        def predict(self, X):
            assert X.dtype == np.float32, X.dtype

            with tf.Session() as s:
                return s.run([self.classif], feed_dict={self.x_ph: X})
    """

    classifiers = [
                   Feedforward([None, linear_X.shape[1]], [None, 2]),  # The first None is the Number of minibatches. .. The second one too.
                 #  dnn,
                   SVC(tol=1e-4, kernel="poly"),
                   LinearSVC(),
                   logistic.LogisticRegression(),
                   RandomForestClassifier(n_estimators=10),
                   KNeighborsClassifier(),
                   ]

    one_hot_set = {tflearn.DNN, Feedforward}
    from collections import Counter
    assert np.std(linear_X) > 1E-4


    for classifier in classifiers:
        if type(classifier) in one_hot_set:
            features_tr = linear_X[train_picks, :]
            labels_tr = to_one_hot(linear_Y[train_picks], 2)
            assert np.std(features_tr) > 1E-3
            assert np.std(labels_tr) > 1E-3

            features_va = linear_X[valid_picks, :]
            labels_va = to_one_hot(linear_Y[valid_picks], 2)
            assert np.std(features_va) > 1E-3
            assert np.std(labels_va) > 1E-3


            classifier.fit(features_tr, labels_tr, n_epoch=1000) #, validation_set=(features_va, labels_va))


            if type(classifier) == tflearn.DNN:
                predicted_va = np.argmax(classifier.predict(features_va), axis=1)
                print("-------------------------------------")
                print("mean: {}".format(                       np.mean(predicted_va == labels_va)        ))
                print("valid predicted Counter: {}".format(    Counter(predicted_va)                     ))
                print("valid labels Counter: {}".format(       Counter(linear_Y[valid_picks].tolist())   ))
                print("-------------------------------------")

        else:
            features_tr = linear_X[train_picks, :]
            labels_tr = linear_Y[train_picks]

            features_va = linear_X[valid_picks]
            labels_va = linear_Y[valid_picks]

            cl = classifier.fit(features_tr, labels_tr)

            print("-------------------------------------")
            print("classifier: {}".format(classifier))
            print("score: {}".format(cl.score(features_va, labels_va)))
            print("avg: {}".format(np.average(cl.predict(features_va))))
            print("-------------------------------------")


def spatial_classification(interp_X, interp_Y, train_picks, valid_picks, test_picks):
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

def make_interpolated_data(X, Y, min_width, max_width, min_height, max_height, res, method):

    grid = np.mgrid[min_width:max_width:res[0]], np.mgrid[min_height:max_height:res[0]]
    interp_X = []
    for i in xrange(X.shape[X_Dims.fft_ch.value]):
        interp_X.append(scipy.interp.griddata(X, Y, grid, method))

    interp_X = np.vstack(interp_X)

    return interp_X

def main(argv):
    start = time.time()

    X, Y = maybe_prep_psds(argv) # argv being passed is temporary

    print("")
    print(np.average(X))
    std_dev = np.std(X)
    print(std_dev)
    print("")

    ###########################################################################
    # CLASSICAL MACHINE LEARNING CLASSIFICATION without locality
    ###########################################################################

    training_picks, valid_picks, test_picks = make_picks(X.shape[0])

    reg_print("# CLASSICAL MACHINE LEARNING")
    linear_X, linear_Y = make_samples_linear(X, Y)

    linear_classification(linear_X, linear_Y, training_picks, valid_picks, test_picks)

    # reg_print("# SPATIAL MACHINE LEARNING")
    # interp_X = make_interpolated_data(X, Y, 0, 200, 0, 200, (1000, 1000), "Cubic")
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
    reg_print("TOTAL TIME: {total_time} sec".format(total_time=end - start))
    reg_print("*********************************************************\n"
          "Total: '%s'" % (end - start))

if __name__ == "__main__": main(sys.argv)
