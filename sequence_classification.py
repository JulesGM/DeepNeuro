# -*- coding: utf-8 -*-
#! /usr/bin/env python
from __future__ import division, print_function, with_statement

__author__ = "Jules Gagnon-Marchand"
__credits__ = ["Jules Gagnon-Marchand"]
__license__ = "GPL"
__maintainer__ = "Jules Gagnon-Marchand"
__email__ = "jgagnonmarchand@gmail.com"
__status__ = "Research"

from six import iteritems
from six.moves import xrange, zip as izip
import sys
import os
import numpy as np
import keras


try: import IPython.core.ultratb # ipython like errors
except ImportError: pass  # No IPython. Use default exception printing.
else: sys.excepthook = IPython.core.ultratb.ColorTB()


def sequence_classification(x, y, job_type):

    print(x.shape)
    sys.exit(0)

    batch_size = 16
    epochs = 10
    gru_cells = 32

    x_tr, x_va, x_te = x
    y_tr, y_va, y_te = y

    model = keras.models.Sequential()
    model.add(keras.layers.GRU(gru_cells, input_shape=[]))
    model.add(keras.layers.BatchNormalization())
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_tr, y_tr,  batch_size=batch_size, epochs=epochs)
    score = model.evaluate(x_va, y_va, batch_size=batch_size)
    print("VALID SCORE: %S" % score)