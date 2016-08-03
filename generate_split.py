from __future__ import division, print_function, with_statement
from six.moves import xrange
from six import iteritems

import os
import sys
import random
import json
import numpy as np
import utils.data_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv):
    """
    This script generates training/validation/test split of the input files, and saves them to fif_split.json.
    This script is to be only ran once.
    """

    print("Generating data split json file")

    data_path = argv[1]
    base_path = argv[2] if len(argv) > 2 else os.path.dirname(__file__)

    by_labels = [[], []] # The labels are of value either True or False
    for name, raw, label, total in utils.data_utils.data_gen(data_path):
        by_labels[int(label)].append(name)

    print("")

    for i in xrange(len(by_labels)):
        random.shuffle(by_labels[i]) # 'random.shuffle' shuffles in place

    valid_frac = 0.2
    test_frac = 0.15

    valid_r_c0 = int(np.ceil(valid_frac * len(by_labels[0])))
    test_r_c0 = int(np.ceil(test_frac * len(by_labels[0])))

    valid_split_c0 = max(valid_r_c0, 1)
    test_split_c0 = max(test_r_c0, 1)

    valid_r_c1 = int(np.ceil(valid_frac * len(by_labels[1])))
    test_r_c1 = int(np.ceil(test_frac * len(by_labels[1])))

    valid_split_c1 = max(valid_r_c1, 1) + 1
    test_split_c1 = max(test_r_c1, 1)

    split = {}
    split["training"] = by_labels[0][valid_split_c0 + test_split_c0:]               + by_labels[1][valid_split_c1 + test_split_c1:]
    split["valid"]    = by_labels[0][:valid_split_c0]                               + by_labels[1][:valid_split_c1]
    split["test"]     = by_labels[0][valid_split_c0:valid_split_c0 + test_split_c0] + by_labels[1][valid_split_c1:valid_split_c1 + test_split_c1]

    for k in split.keys():
        random.shuffle(split[k])

    # Inside out
    res = {}
    for k, v in iteritems(split):
        print("{}: {}".format(k, len(v)))
        for name in v:
            assert name not in res, "we should never be trying to add a name that's already in the dict"
            res[name] = k

    json_path = os.path.join(base_path, "fif_split.json")

    with open(json_path, "w") as of:
        json.dump(res, of)

    print("--")

    return 0

if __name__ == "__main__": sys.exit(main(sys.argv))
