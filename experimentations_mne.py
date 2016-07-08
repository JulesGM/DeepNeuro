from __future__ import with_statement, print_function
import mne
import mne.time_frequency
mne.set_log_level('WARNING')

import timeit
import time
import argparse
import sys
import numpy as np


def time_n(f, procs, rep, raw_obj):
    print("---------------------------------------------------------")
    print("f:        '%s'" % f.__name__)
    print("procs:    '%s'" % procs)
    print("rep:      '%s'" % rep)
    stuff = []
    def job():
        stuff.append(f(raw_obj, n_jobs=procs))

    res = np.array(timeit.Timer(job).repeat(rep, 1))
    print("avg:      '%s'" % str(np.average(res)))
    print("min:      '%s'" % str(np.min(res)))
    return res, stuff[0]

def main(argv):
    # defaults
    d_MIN_PROCS = 8
    d_MAX_PROCS = 8
    d_REP = 1
    d_F = "both"

    p = argparse.ArgumentParser(argv)
    p.add_argument("--min_procs", type=int, default=d_MIN_PROCS)
    p.add_argument("--max_procs", type=int, default=d_MAX_PROCS)
    p.add_argument("-r", "--reps", type=int, default=d_REP)
    p.add_argument("-f", "--funcs", type=str, default=d_F)
    args = p.parse_args()

    print("\nArgs:     '%s'" % args)

    MIN_PROCS = args.min_procs
    MAX_PROCS = args.max_procs
    REP = args.reps
    FUNCS = args.funcs

    if FUNCS == "both":
        FUNCS_list = [mne.time_frequency.psd_multitaper, mne.time_frequency.psd_welch]
    elif FUNCS == "welch" or FUNCS == "w":
        FUNCS_list = [mne.time_frequency.psd_welch]
    elif FUNCS == "mt" or FUNCS == "multitaper" or FUNCS == "m":
        FUNCS_list = [mne.time_frequency.psd_multitaper]
    else:
        raise RuntimeError("invalid funcs argument. must be one of ['both', 'w', 'm']. Got '%s'." % FUNCS)


    start = time.clock()
    raw = mne.io.Raw("K0002_rest_raw_tsss_mc_trans_ec_ica_raw.fif")

    actual_min = MIN_PROCS if MIN_PROCS else 1 # 1 if MIN_PROCS is 0 or None

    from collections import defaultdict as dd
    num_res_rec = dd(list)

    for f in FUNCS_list:
        for procs in range(actual_min, MAX_PROCS + 1):
            time_res, num_res = time_n(f, procs, REP, raw)
            num_res_rec[f].append(num_res)

    np.allclose(*[x[0] for x in num_res_rec.values()], atol=1e-4)


    end = time.clock()
    print("*********************************************************\nTotal: '%s'" % (end - start))

if __name__ == "__main__": main(sys.argv)