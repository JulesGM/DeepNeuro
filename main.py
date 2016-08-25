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
ARGS_ONLY_NAME = "args_only"

@click.group(invoke_without_command=True)
@click.option("--nfft",               type=int,     default=100)
@click.option("--fmax",               type=int,     default=200)
@click.option("--tincr",              type=float,   default=1)
@click.option("--established_bands",                default="quarter")
@click.option("--limit",              type=int,     default=None)
@click.option("--tmin",               type=int,     default=0)
@click.option("--tmax",               type=int,     default=1000000)
@click.option("--sensor_type",        type=str,     default="grad")
@click.option("--noverlap",           type=int,     default=0)
@click.option("--data_path",          type=str,     default=os.path.join(os.environ["HOME"], "aut_gamma"))
@click.option("--" + ARGS_ONLY_NAME,  type=bool,    default=False, is_flag=True)
@click.pass_context
def main(ctx, **click_options):
    if ctx.invoked_subcommand is None:
        print("############################################################################################\n"
              "\n   Warning : \n"
              "\tThe program was called without a command.\n\n"
              "\tCalling this program without a command only prepares the PSD data.\n\n"
              "\tUse --help to list the available commands.\n\n"
              "############################################################################################")

    print("\nArgs:")
    utils.print_args(None, click_options)

    ctx.obj["main"] = click_options
    if click_options[ARGS_ONLY_NAME]:
        return

    if six.PY3:
        print("The code hasn't been tested in Python 3.\n")

    json_split_path = os.path.join(base_path, "fif_split.json")

    if not os.path.exists(json_split_path):
        raise RuntimeError("Couldn't find fif_split.json. Should be generated with ./generate_split.py at the beginning"
                           " of the data exploration, and then shared.")

    # This part would be much cleaner without passing the `args` object to `maybe_prep_psds`.
    # We haven't found the time to do this conversion yet.
    from argparse import Namespace
    args = Namespace(**click_options)
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


@main.command(help="- Linear classification")
@click.argument("job_type",             type=str,         default="SVM")
@click.pass_context
def lc(ctx, job_type):
    print("Spatial Classification args:")
    click_positional = {"job_type": job_type}
    utils.print_args(click_positional, None)
    if ctx.obj["main"][ARGS_ONLY_NAME]:
        return

    # We put the imports to classification managers inside of the function to not trigger
    # the very slow import of tensorflow even when just showing the help text, for example
    import linear_classification
    linear_classification.linear_classification(ctx.obj["main"]["X"], ctx.obj["main"]["Y"], job_type)


@main.command(help="- Spatial classification")
@click.argument("net_type",             default="tflearn_resnet",      type=str)
@click.option("--res",                  default=(224, 224), type=(int, int)) # to match vgg_cifar
@click.option("--dropout_keep_prob",    default=1,          type=float)
@click.option("--learning_rate",        default=0.001,      type=float)
@click.option("--depth",                default=7,          type=int)
@click.option("--minibatch_size",       default=256,        type=int)
@click.option("--filter_scale_factor",  default=2,          type=float)
@click.option("--dry_run",              default=False,      type=bool, is_flag=True)
@click.option("--test_qty",             default=2048,       type=int)
@click.pass_context
def sc(ctx, net_type, **click_options):
    # As it it quite common to leave quite a few options at their default values,
    # it is helpful to print all the values to make it less likely a default has an unexpected value.
    print("Spatial Classification args:")
    click_positional = {"net_type": net_type}
    utils.print_args(click_positional, click_options)
    if ctx.obj["main"][ARGS_ONLY_NAME]:
        return

    #click_options["sensor_type"] = True if click_options["sensor_type"] == "both" else click_options["sensor_type"]
    from_main = ctx.obj["main"]

    # We put the imports to classification managers inside of the function to not trigger
    # the very slow import of tensorflow even when just showing the help text, for example
    import spatial_classification
    spatial_classification.spatial_classification(
        from_main["X"], from_main["Y"], sensor_type=from_main["sensor_type"], nfft=from_main["nfft"], tincr=from_main["tincr"], fmax=from_main["fmax"],
        info=from_main["info"], established_bands=from_main["established_bands"], net_type=net_type, **click_options)

if __name__ == "__main__": main(obj={})
