#! /usr/bin/env python

from __future__ import print_function, division
from six.moves import xrange
from six import iteritems
import argparse, sys, os, subprocess as sp

import json
"""

This script is meant to be run on the gateway computer.
It's the one that starts the actual jobs.

"""
def main(argv):
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################

    BASE_PATH = os.path.dirname(__file__)

    json_config_path = os.path.join(BASE_PATH, "gw_args.json")
    if len(argv) <= 1:
        if os.path.exists(json_config_path):
            with open(json_config_path, "r") as _if:
                argv = json.load(_if)
                print(argv)
        else:
            raise RuntimeError("We have neither args or a gw_args.json")

    else:
        with open(json_config_path, "w") as of:
            json.dump(argv, of)

    p = argparse.ArgumentParser()
    p.add_argument("--host",        type=str)
    p.add_argument("--data_path",   type=str)
    p.add_argument("--job_type",    type=str)
    args = p.parse_args(argv[1:])

    ###########################################################################
    # CONFIG
    ###########################################################################
    SCRIPT_PATH = os.path.join(BASE_PATH, "msub_PSD_perf_exploration.sh")
    walltime = "10:00:00" # this is the max we are allowed

    #
    tincr_constant = 10
    tincr_min_exp = 1
    tincr_max_exp = 1#3
    tincr_incr_exp = 1


    nfft_constant = 10
    nfft_min_exp = 1
    nfft_max_exp = 1#3
    nfft_exp_incr = 1

    # Currently, novelap == 0
    noverlap_min = 0
    noverlap_max = 0
    noverlap_incr = 10

    JOB_QTY_LIMIT = 1001
    total_jobs = ((tincr_max_exp - tincr_min_exp + 1) // tincr_incr_exp) * ((noverlap_max - noverlap_min + 1) // noverlap_incr) * (
        (nfft_max_exp - nfft_min_exp + 1) // nfft_exp_incr)

    print("Total jobs: {}".format(total_jobs))

    assert total_jobs < JOB_QTY_LIMIT, \
            "total_jobs >= JOB_QTY_LIMIT; got total_jobs=={total_jobs}, job_qty_limit=={JOB_QTY_LIMIT}."\
            "This is probably too much.".format(job_qty_limit=JOB_QTY_LIMIT, total_jobs=total_jobs)

    ###########################################################################
    # RUN THE JOBS
    ###########################################################################
    # Do the grid search by running a bunch of jobs on the cluster with different variable ranges

    launcher_args = {}
    for tincr_exp in xrange(tincr_min_exp, tincr_max_exp + 1, tincr_incr_exp):
        for nfft_exp in xrange(nfft_min_exp, nfft_max_exp + 1, nfft_exp_incr):
            for noverlap in xrange(noverlap_min, noverlap_max + 1, noverlap_incr):
                # Test 2 : noverlap < nfft // 2

                launcher_args["--nfft"] =       nfft_constant ** nfft_exp
                launcher_args["--glob_tmin"] =  0
                launcher_args["--glob_tincr"] = tincr_constant**tincr_exp
                launcher_args["--noverlap"] =   noverlap
                launcher_args["-o"] =           args.data_path
                launcher_args["--job_type"] =   args.job_type
                assert len(
                    launcher_args) == 6, "Meant to be a proof that we change the whole dict every inner loop. Got %s, should've gotten 5." % len(
                    launcher_args)

                # The script is only loading appropriate modules and forwarding a bunch of args to a python script.
                # We prepare the arguments so they are easy to forward.
                launcher_args_str = "\"%s\"" % " ".join(["{k} {v}".format(k=k, v=v) for k, v in iteritems(launcher_args)])
                print(SCRIPT_PATH)
                print("exists: %s" % os.path.exists(SCRIPT_PATH))


                # We preallocate spaces for the msub qsub launcher string and for the argument passing solution at 8 & 9
                command = [
                    None, SCRIPT_PATH,
                    "-l", "nodes=1:gpus=1:ppn=1",
                    "-l", "walltime={walltime}".format(walltime=walltime),
                    "-A", "kif-392-aa",
                    None, None,
                ]

                # The cluster unit specific config is here

                if args.host == "guillimin":
                    print(">>>>>>> THE GUILLIMIN LAUNCH SCRIPT HASN'T BEEN TESTED IN AGES <<<<<<<")
                    command[0] = "qsub" # the job submitting binary
                    # A way to forward args.
                    # command[8] = "-F"
                    # command[9] = launcher_args_str

                elif args.host == "helios":
                    command[0] = "/opt/moab/bin/msub" # the job submitting binary
                    # A way to forward args.

                else:
                    raise RuntimeError("Unrecognized host. Got '%s'" % args.host)

                command[8] = "-v"
                command[9] = "ARGS=%s" % launcher_args_str

                print("launching following code from gateway node:\n%s" % command)



                print(sp.check_output(command))

if __name__ == "__main__":
    main(sys.argv)