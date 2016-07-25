#! /usr/bin/env python

from __future__ import print_function, division
from six.moves import xrange
from six import iteritems
import argparse, sys, os, subprocess as sp

import json
"""
http://martinos.org/mne/stable/generated/mne.time_frequency.psd_welch.html#mne.time_frequency.psd_welch
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
    args = p.parse_args(argv[1:])

    ###########################################################################
    # CONFIG
    ###########################################################################
    SCRIPT_PATH = os.path.join(BASE_PATH, "msub_PSD_perf_exploration.sh")
    walltime = "10:00:00" # this is the max we are allowed

    tincr_min = 10
    tincr_max = 10
    tincr_incr = 1

    nfft_min = 10
    nfft_max = 80
    nfft_incr = 100

    noverlap_min = 0
    noverlap_max = 0
    noverlap_incr = 10

    JOB_QTY_LIMIT = 1001
    total_jobs = ((tincr_max - tincr_min) // tincr_incr) * ((noverlap_max - noverlap_min) // noverlap_incr) * (
        (nfft_max - nfft_min) // nfft_incr)
    print("Total jobs: {}".format(total_jobs))

    assert total_jobs < JOB_QTY_LIMIT, \
            "total_jobs >= JOB_QTY_LIMIT; got total_jobs=={total_jobs}, job_qty_limit=={JOB_QTY_LIMIT}."\
            "This is probably too much.".format(job_qty_limit=JOB_QTY_LIMIT, total_jobs=total_jobs)

    ###########################################################################
    # RUN THE JOBS
    ###########################################################################
    # Do the grid search by running a bunch of jobs on the cluster with different variable ranges

    launcher_args = {}
    for tincr in xrange(tincr_min, tincr_max + 1, tincr_incr):
        for nfft in xrange(nfft_min, nfft_max + 1, nfft_incr):
            for noverlap in xrange(noverlap_min, noverlap_max + 1, noverlap_incr):
                # Test 2 : noverlap < nfft // 2
                try:
                    assert noverlap <= nfft // 2, "noverlap > nfft // 2; got noverlap=={noverlap} and nfft=={nfft}." \
                    "THIS COMBINATION OF VALUES WILL BE IGNORED, BUT THE SCRIPT CONTINUES."\
                        .format(noverlap=noverlap, nfft=nfft).replace("\t", "")

                except AssertionError, err:
                    print(err)
                    continue

                launcher_args["--nfft"] =       nfft
                launcher_args["--glob_tmin"] =  0
                # launcher_args["--glob_tmax"] =  None
                launcher_args["--glob_tincr"] = tincr
                launcher_args["--noverlap"] =   noverlap
                launcher_args["-o"] =           args.data_path
                assert len(
                    launcher_args) == 5, "Meant to be a proof that we change the whole dict every inner loop. Got %s, should've gotten 5." % len(
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