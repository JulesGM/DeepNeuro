from __future__ import print_function, division, with_statement

import numpy as np
import enum
import six


class X_Dims(enum.Enum):
    """ The use of this is optional. It is only meant to make the understanding of the dims easier to understand.
    """
    samples_and_times = 0
    fft_ch = 1
    sensors = 2
    size = 3


def make_samples_linear(x):
    """ So, by 'making samples linear', we mean that we flatten everything so that we get a large 1d vector 
        composed of all of the channels of each of the sensors.
    """
    target_shape = (x.shape[X_Dims.samples_and_times.value], x.shape[X_Dims.fft_ch.value] * x.shape[X_Dims.sensors.value])
    print("target_shape: {}".format(target_shape))
    linear_x = x.reshape(*target_shape)
    return linear_x
    

def to_one_hot(input, max_classes):
    no_samples = input.shape[0]
    output = np.zeros((input.shape[0], int(max_classes)), np.float32)
    output[np.arange(no_samples), input.astype(np.int32)] = 1
    return output


def from_one_hot(values):
    return np.argmax(values, axis=1)


def _print_dict_sorted(dict_):
    # aux func
    for k, w in sorted(six.iteritems(dict_), key=lambda k_v_pair: k_v_pair[0]):
        print("\t - {:20} {}".format(str(k) + ":", w))


def print_args(positional, named):
    if positional:
        print("\tPositional:")
        _print_dict_sorted(positional)
    else:
        print("\t[No positional arguments]")
    print("")
    if named:
        print("\tNamed:")
        _print_dict_sorted(named)
    else:
        print("\t[No named options]")
    print("")
