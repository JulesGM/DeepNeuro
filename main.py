#! /usr/bin/env python
# compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

# stdlib imports
import os
import logging

# local imports
import utils
import utils.data_utils
import linear_classification
import spatial_classification

# external imports
import numpy as np
import click

"""
MNE's logger prints massive amount of useless stuff, and neither mne.set_logging_level(logging.ERROR) or
logging.basicConfig(level=logging.ERROR) seem to be working.
"""
logger = logging.getLogger('mne')
logger.disabled = True
base_path = os.path.dirname(__file__)

@click.group()
@click.option("--nfft",               type=int,     default=3000)
@click.option("--fmax",               type=int,     default=100)
@click.option("--tincr",              type=float,   default=1)
@click.option("--established_bands",  type=bool,    default=False)
@click.option("--limit",              type=int,     default=None)
@click.option("--tmin",               type=int,     default=0)
@click.option("--tmax",               type=int,     default=1000000)
@click.option("--noverlap",           type=int,     default=0)
@click.option("--data_path",          type=str,     default=os.path.join(os.environ["HOME"], "aut_gamma"))
@click.pass_context
def main(ctx, **kwargs):
    ctx.obj["main"] = kwargs

    if six.PY3:
        print("The code hasn't been tested in Python 3.\n")

    print("\nArgs:")
    for key, value in sorted(six.iteritems(ctx.obj["main"]), key=lambda k_v_pair: k_v_pair[0]):
        print("\t- {:12}: {}".format(key, value))
    print("--")

    json_split_path = os.path.join(base_path, "fif_split.json")

    if not os.path.exists(json_split_path):
        raise RuntimeError("Couldn't find fif_split.json. Should be generated with ./generate_split.py at the beginning"
                           " of the data exploration, and then shared.")

    from argparse import Namespace
    args = Namespace(**ctx.obj["main"])
    X, Y, sample_info = utils.data_utils.maybe_prep_psds(args)

    ctx.obj["main"]["X"] = X
    ctx.obj["main"]["Y"] = Y
    ctx.obj["main"]["info"] = sample_info

    print("Dataset properties:")
    for i in xrange(3):
        print("\t- {} nan/inf:    {}".format(i, np.any(np.isnan(X[i]))))
        print("\t- {} shape:      {}".format(i, X[i].shape))
        print("\t- {} mean:       {}".format(i, np.mean(X[i])))
        print("\t- {} stddev:     {}".format(i, np.std(X[i])))
        print("\t--")
    print("--")


@main.command()
@click.argument("job_type",             type=str,         default="SVM")
@click.pass_context
def lc(ctx, job_type):
    linear_classification.linear_classification(ctx.obj["main"]["X"], ctx.obj["main"]["Y"], job_type)


@main.command()
@click.argument("net_type",             default="cnn",    type=str)
@click.option("--res",                  default=(33, 33), type=(int, int))
@click.option("--dropout_keep_prob",    default=0.9,      type=float)
@click.option("--learning_rate",        default=0.001,    type=float)
@click.option("--depth",                default=9,        type=int)
@click.option("--minibatch_size",       default=128,      type=int)
@click.option("--sensor_type",          default="both",   type=str)
@click.option("--filter_scale_factor",  default=1,        type=float)
@click.pass_context
def sc(ctx, net_type, **kwargs):
    kwargs["sensor_type"] = True if kwargs["sensor_type"] == "both" else kwargs["sensor_type"]
    from_main = ctx.obj["main"] # easier to read
    spatial_classification.spatial_classification(
        from_main["X"], from_main["Y"], nfft=from_main["nfft"], tincr=from_main["tincr"], fmax=from_main["fmax"],
        info=from_main["info"], established_bands=from_main["established_bands"], net_type=net_type, **kwargs)

if __name__ == "__main__": main(obj={})
