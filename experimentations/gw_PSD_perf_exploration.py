from __future__ import print_function, division
from six.moves import xrange
from six import iteritems
import argparse, sys, os, subprocess as sp


"""
http://martinos.org/mne/stable/generated/mne.time_frequency.psd_welch.html#mne.time_frequency.psd_welch
"""
def main(argv):
    ###########################################################################
    # ARGUMENT PARSING
    ###########################################################################
    p = argparse.ArgumentParser()
    p.add_argument("--launcher",    type=str)
    p.add_argument("--host",        type=str)
    p.add_argument("--data_path",   type=str)
    args = p.parse_args(argv[1:])

    assert args.launcher in ["msub", "qsub"], "got %s" % args.launcher

    ###########################################################################
    # CONFIG
    ###########################################################################
    SCRIPT_PATH = "/home/julesgm/COCO/experimentations/msub_PSD_perf_exploration.sh"
    REPS = 1
    walltime="10:00:00"

    tincr_min  = 10
    tincr_max  = 10
    tincr_incr = 1

    nfft_min  = 1000
    nfft_max  = 1000
    nfft_incr = 100

    noverlap_min  = 0
    noverlap_max  = 0
    noverlap_incr = 10

    ###
    # Verify that we aren't going crazy with the quantity of jobs
    # 1000 jobs * .25 hours = 250 hours ~= 10 days with .25 hour granularity for 8 procs
    #                       = 10 / 8 = 1.25 days = 30 h
    ##

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
                launcher_args["--reps"] =       REPS
                launcher_args["-o"] =           args.data_path
                assert len(
                    launcher_args) == 6, "Meant to be a proof that we change the whole dict every inner loop. Got %s, should've gotten 6." % len(
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

                if args.host == "guillimin":
                    command[0] = "qsub"
                    command[8] = "-F"
                    command[9] = launcher_args_str

                elif args.host == "helios":
                    command[0] = "/opt/moab/bin/msub"
                    command[8] = "-v"
                    command[9] = "ARGS=%s" % launcher_args_str

                else:
                    raise RuntimeError("Unrecognized host. Got '%s'" % args.host)

                print("launching following code from gateway node:\n%s" % command)
                print(sp.check_output(command))

if __name__ == "__main__":
    main(sys.argv)