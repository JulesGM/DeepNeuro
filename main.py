from __future__ import division, print_function, with_statement
range = xrange
import matplotlib as mpl
import tensorflow as tf

import sys, os, re, fnmatch, subprocess as sp, argparse as ap, logging, threading
from collections import defaultdict
import mne
mne.set_config("MNE_USE_CUDA", "true")
mne.cuda.init_cuda()

import mne.io.pick



from data_utils import *
from utils import *
from numpy.random import permutation



def indice_gen(no_samples):
    while True:
        indices = permutation(np.arange(no_samples))
        for index in indices:
            yield index


def gen_data(features, labels, minibatch_size):
    assert features.shape[0] == labels.shape[0]
    no_samples = features.shape[0]
    indices_gen_inst = indice_gen(no_samples)

    while True:
        indices_minibatch = np.array([next(indices_gen_inst) for _ in range(minibatch_size)], dtype=np.int64)
        feature, label = features[indices_minibatch, :, :, :], label[indices_minibatch]
        yield feature, label


def train(net, y_placeholder, x_values, y_values, minibatch_size, max_epochs = 10):
    loss = tf.nn.softmax_cross_entropy_with_logits(net, y_placeholder)
    optim = tf.train.AdamOptimizer().minimize(loss)
    gen_data_inst = gen_data(x_values, y_values, minibatch_size)
    no_samples = x_values.shape[0]

    with tf.Session() as s:
        for i in range(max_epochs):
            for j in range(no_samples): # each epoch is a full iteration on the data

                x, y = next(gen_data_inst)


                opt, loss = s.run((optim, loss), feed_dict= dict(
                    x=x,
                    y=y))


                count = j + i * no_samples
                if count % 100 == 0 and count != 0:
                    print(loss)


def resnet_0(x, y):
    from custom_cells.resnet_0 import res_net
    return res_net(x, y)

"""
def resnet_1(x, y):
    # https: // github.com / ry / tensorflow - resnet / blob / master / resnet.py
    from custom_cells.resnet_1 import resnet, resnet_train
    logits = resnet.inference_small(x, True, num_blocks=3, use_bias=False, num_classes=2)
    return resnet_train.train(True, logits, x, y)
"""

def convnet_0(x_shape, minibatch_size):
    """
    Super simple test cnn
    :param x:
    :param y:
    :return:
    """
    all_conv = dict(
        padding="SAME",
        use_cudnn_on_gpu=True)

    conv0_output_shape = None
    conv1_output_shape = None

    pool         = 3
    pool_s       = 3
    pool_ksize   = [1, pool, pool, 1]
    pool_strides = [1, pool_s, pool_s, 1]
    all_pool = dict(
        padding="SAME",
        ksize=pool_ksize,
        strides=pool_strides
    )

    CONV0_w_shape      = np.zeros([4], dtype=np.float32)
    CONV0_w_shape[0]   = minibatch_size
    CONV0_w_shape[1:]  = x_shape[1:]
    CONV0_w_shape[3]  += 16

    CONV1_w_shape      = np.zeros([4], dtype=np.float32)
    CONV1_w_shape[0]   = CONV0_w_shape[0]
    CONV1_w_shape[1:2] = CONV0_w_shape[1:2] / pool
    CONV1_w_shape[3]   = CONV1_w_shape[1:2] + np.ones_like(CONV1_w_shape[1:2]) * 16

    X = tf.placeholder(tf.float32, [minibatch_size, x_shape[1], x_shape[2], x_shape[3]])

    CONV0_w = tf.Variable(initial_value=tf.truncated_normal((x_shape, conv0_output_shape)))
    net = tf.nn.conv2d(X, CONV0_w, **all_conv)
    net = tf.nn.relu(tf.matmul(net, CONV0_w))
    net = tf.nn.max_pool(net, **all_pool)

    CONV1_w = tf.Variable(initial_value=tf.truncated_normal((net.get_shape(), conv1_output_shape)))
    net = tf.nn.conv2d(net, CONV1_w, **all_conv)
    net = tf.nn.relu(tf.matmul(net, CONV1_w))
    net = tf.nn.max_pool(net, **all_pool)

    FC0_w = tf.Variable(initial_value=tf.truncated_normal(()))
    net = tf.nn.softmax(tf.matmul(net, FC0_w))

    return net


def main(argv):
    targets = ["K0002_rest_raw_tsss_mc_trans_ec_ica_raw.fif"]

    ###################################################################
    # Misc settings
    base_path = os.path.dirname(os.path.abspath(__file__))
    mne.set_log_file("./mne_log.log", overwrite=True)
    ###################################################################


    ###################################################################
    # Argument parsing
    defaults = dict(
        data_path ="/media/jules/JULES700GB/Dropbox/aut_gamma/MEG",
        hdf5_path = "/media/jules/JULES700GB/COCOLAB/data.hdf5"
    )
    # args = parse_args(argv, defaults)

    if len(argv) == 3:
        print(">> argv :: %s" % str(argv))
        proc_no  = int(argv[1])
        interval = int(argv[2])
        print(">> Multi_proc :: proc %s started" % proc_no)

    else:
        proc_no = None
        interval = 100

    ###################################################################


    ###################################################################
    # Generic data prep
    """
    no_features = len(targets)
    features = [] # to be converted to numpy eventually
    labels = np.empty(no_features, np.bool)
    for target_name in targets:
        target = os.path.join(base_path, target_name)
        raw = Raw(target)
        label = target_name.startswith("K0")
        frames = make_arrays(uperrange, raw, proc_no, interval,
                        base_path, meg_type, ANIM, 10, 400)
        features.append(frames)
        labels.append(label)

    """
    ###################################################################


    ###################################################################
    # Steps:
    #   1.  Extract Raw data from the files
    #   2.  Build topographical maps
    #   3.  Define Model
    #   4.  Train
    #       .1  .. partial epoch
    #       .2  .. save & test
    #       .3  .. continue
    #   5.  ... Done

    minibatch_size = 64
    x_axis_length = 64
    y_axis_length = 64
    no_channels = 1


#   X = tf.placeholder(dtype=np.float32, shape=(minibatch_size, x_axis_length, y_axis_length, no_channels))
    Y = tf.placeholder(dtype=np.float32, shape=(minibatch_size, ))
    x_np = make_arrays(10, Raw(targets[0]), None, 10, base_path, "mag", False)

    x_np = x_np.reshape(list(x_np.shape) + [1]).astype(np.float32)


    x = tf.constant(x_np, dtype=tf.float32, shape=x_np.shape)
    y_np = np.random.binomial(1, 0.5, x_np.shape[0]).astype(np.float32)

    y = tf.constant(y_np, dtype=tf.float32, shape=y_np.shape)

    print(x_np.shape)

    train(convnet_0(x_np.shape, minibatch_size), Y, x, y, 64)

    return 0



if __name__ == "__main__": sys.exit(main(sys.argv))