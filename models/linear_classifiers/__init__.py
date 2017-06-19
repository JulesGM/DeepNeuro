from __future__ import with_statement, print_function, division, absolute_import
import sys, os
sys.path.append(os.path.dirname(__file__))
from utils import X_Dims

def make_samples_linear(x):
    target_shape = (x.shape[X_Dims.samples_and_times.value], x.shape[X_Dims.fft_ch.value] * x.shape[X_Dims.sensors.value])
    print("target_shape: {}".format(target_shape))
    linear_x = x.reshape(*target_shape)
    return linear_x