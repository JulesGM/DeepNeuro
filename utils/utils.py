from __future__ import print_function, division, with_statement

import numpy as np
import enum

class X_Dims(enum.Enum):
    # Meh
    samples_and_times = 0
    fft_ch = 1
    sensors = 2
    size = 3

def to_one_hot(input, max_classes):
    no_samples = input.shape[0]
    output = np.zeros((input.shape[0], int(max_classes)), np.float32)
    output[np.arange(no_samples), input.astype(np.int32)] = 1
    return output

def from_one_hot(values):
    return np.argmax(values, axis=1)
