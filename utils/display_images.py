#! /usr/bin/env python
from __future__ import division, print_function, with_statement
from six import iteritems
from six.moves import xrange, zip as izip
import sys
import os

import utils

import numpy as np
import numpy.random
import click
import h5py
from matplotlib import pyplot as plt


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--no", type=int, default=10)
def main(path, no):

    hdf5_saver_loader = utils.data_utils.HDF5SaverLoader(path)
    images = hdf5_saver_loader.load_ds()

    final_image = np.hstack((im[np.random.randint(0, im.shape[0], no), :, :, :].reshape(im.shape[1], im.shape[0] * im.shape[2], None) for im in images))



    return 0


if __name__ == "__main__":
    sys.exit(main())