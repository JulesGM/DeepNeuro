#! /usr/bin/env python
# -*- coding: utf-8 -*-
# compatibility imports
from __future__ import with_statement, print_function, division
import six
from six.moves import range as xrange
from six.moves import zip as izip

# stdlib imports
import os
import logging
from argparse import Namespace

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
base_path = os.path.dirname(os.path.realpath(__file__))


@click.group(invoke_without_command=True)
@click.option("--nfft",               type=int,     default=1000) # 1000 for Quarter Established_bands
@click.option("--fmax",               type=int,     default=200)
@click.option("--tincr",              type=float,   default=1)
@click.option("--established_bands",                default="quarter") # True, False, "quarter", "half"
@click.option("--limit",              type=int,     default=None)
@click.option("--tmin",               type=int,     default=0)
@click.option("--tmax",               type=int,     default=1000000)
@click.option("--sensor_type",        type=str,     default=True)
@click.option("--noverlap",           type=int,     default=0)
@click.option("--data_path",          type=str,     default=os.path.join(os.environ["HOME"], "aut_gamma"))
@click.option("--is_time_dependant",  is_flag=True)
@click.option("--args_only",          is_flag=True)
@click.pass_context
def main(ctx, **args):
    """ This part prepares the data.

    """
    if ctx.invoked_subcommand is None:
        print("############################################################################################\n"
              "\n   Warning : \n"
              "\tThe program was called without a command.\n\n"
              "\tCalling this program without a command only prepares the PSD data.\n\n"
              "\tUse --help to list the available commands.\n\n"
              "############################################################################################")

    print("\nArgs:")
    utils.print_args(None, args)
    ctx.obj["main"] = args
    args = Namespace(**args) # Namespaces are easier to manipulate

    if args.args_only:
        return

    if six.PY3:
        print("This code hasn't been tested in Python 3.\n")

    json_split_path = os.path.join(base_path, "fif_split.json")

    if not os.path.exists(json_split_path):
        raise RuntimeError("Couldn't find fif_split.json. Should be generated with ./generate_split.py"
                           "at the beginning of the data exploration, and then shared.")

    x, y, sample_info = utils.data_utils.maybe_prep_psds(args)

    ctx.obj["main"]["x"] = x
    ctx.obj["main"]["y"] = y
    ctx.obj["main"]["info"] = sample_info

    print("Dataset properties:")
    if args.is_time_dependant:
        pass
    else:
        for i in xrange(3):
            print("\t- {} nan/inf:    {}".format(i, np.any(np.isnan(x[i]))))
            print("\t- {} shape:      {}".format(i, x[i].shape))
            print("\t- {} mean:       {}".format(i, np.mean(x[i])))
            print("\t- {} stddev:     {}".format(i, np.std(x[i])))
            print("\t--")
        print("--")


##
# The different models each have a different point of entry

@main.command(help="- single dimension classification")
@click.pass_context
def SVM_rbf(ctx):
    from models.linear_classifiers.SVM_rbf import experiment
    experiment(ctx.obj["main"]["x"], ctx.obj["main"]["y"])


@main.command(help="- single dimension classification")
@click.pass_context
def SVM_linear(ctx):
    from models.linear_classifiers.SVM_linear import experiment
    experiment(ctx.obj["main"]["x"], ctx.obj["main"]["y"])


@main.command(help="- Point based spatial classification")
@click.pass_context
def point_based(ctx):
    from models.spatial_classifiers.point_based.modified_pointnet_train import experiment
    experiment(ctx.obj["main"]["x"], ctx.obj["main"]["y"], ctx.obj["main"]["info"])


@main.command(help="- Sequence classification")
@click.argument("job_type", type=str, nargs=-1)
@click.pass_context
def sequence(ctx, job_type):
    assert False, "NOT FUNCTIONAL"

    print("Sequence Classification args:")
    click_positional = {"job_type": job_type}
    utils.print_args(click_positional, None)
    if ctx.obj["main"]["args_only"]:
        return

    assert ctx.obj["main"]["is_time_dependant"]

    # We put the imports to classification managers inside of the function to not trigger
    # the very slow import of tensorflow even when just showing the help text, for example
    import sequence_classification
    sequence_classification.sequence_classification(ctx.obj["main"]["x"], ctx.obj["main"]["y"], job_type)


@main.command(help="- Spatial classification")
@click.argument("net_type",             default="conv",     type=str)
@click.option("--res",                  default=(25, 25),     type=(int, int)) # to match vgg_cifar
@click.option("--dropout_keep_prob",    default=0.50,         type=float)
@click.option("--learning_rate",        default=0.0002,       type=float)
@click.option("--minibatch_size",       default=1024,         type=int)
@click.option("--dry_run",              default=False,        type=bool, is_flag=True)
@click.option("--test_qty",             default=2048,         type=int)
@click.option("--load_model",           default=None,)
@click.option("--model_save_path",      default=None,)
@click.pass_context
def conv(ctx, net_type, **args):
    # As it it quite common to leave quite a few options at their default values,
    # it is helpful to print all the values to make it less likely a default has an unexpected value.
    print("Spatial Classification args:")
    click_positional = {"net_type": net_type}
    utils.print_args(click_positional, args)
    if ctx.obj["main"]["args_only"]:
        return

    # Breaks encapsulation. However, this is research code, and this allows for very quick evolution of the argument
    # scheme. This is not production code.
    args.update(ctx.obj["main"])
    args.update(click_positional)
    args = Namespace(**args) # Namespaces are easier to manipulate

    # Soft verification that we are using the time dependant mode of the data generation if we are using RNNs.
    # The soft part is knowing whether we are actually using RNNs of course
    lowered_net_type = args.net_type.lower()
    if "rnn" in lowered_net_type or "lstm" in lowered_net_type or "time" in lowered_net_type:
        assert args.is_time_dependant

    # We put the imports to classification managers inside of the function to not trigger
    # the very slow import of tensorflow even when just showing the help text, for example
    import spatial_classification
    spatial_classification.spatial_classification(args)

if __name__ == "__main__": main(obj={})
