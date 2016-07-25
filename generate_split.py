from __future__ import division, print_function, with_statement
from six.moves import xrange
from six import iteritems
import os, sys, random, json
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import *

from collections import defaultdict

def main(argv):
    """
    This script generates training/validation/test split of the input files, and saves them to fif_split.json.
    This script is to be only ran once. Really.
    """

    print("************************GENERATE_SPLIT************************")

    DATA_PATH = argv[1]
    BASE_PATH = argv[2] if len(argv) > 2 else os.path.dirname(__file__)

    print("DATA_PATH: {}".format(DATA_PATH))
    print("BASE_PATH: {}".format(BASE_PATH))

    by_labels = [[], []] # The labels are of value either True or False
    for name, raw, label in data_gen(DATA_PATH):
        by_labels[int(label)].append(name)

    print("")

    for i in xrange(len(by_labels)):
        random.shuffle(by_labels[i]) # 'random.shuffle' shuffles in place

    valid_r_c0 = int(0.2 * len(by_labels[0]))
    test_r_c0 = int(0.1 * len(by_labels[0]))

    valid_split_c0 = max(valid_r_c0, 1)
    test_split_c0 = max(test_r_c0, 1)

    valid_r_c1 = int(0.2 * len(by_labels[1]))
    test_r_c1 = int(0.1 * len(by_labels[1]))

    valid_split_c1 = max(valid_r_c1, 1)
    test_split_c1 = max(test_r_c1, 1)

    split = {}
    split["training"] = by_labels[0][valid_split_c0 + test_split_c0:] + by_labels[1][valid_split_c1 + test_split_c1:]
    split["valid"] = by_labels[0][valid_split_c0:valid_split_c0 + test_split_c0] + by_labels[1][valid_split_c1:valid_split_c1 + test_split_c1]
    split["test"] = by_labels[0][0:valid_split_c0] + by_labels[1][0:valid_split_c1]

    for k in split.keys():
        random.shuffle(split[k])

    # Inside out
    res = {}
    for k, v in iteritems(split):
        print("{}: {}".format(k, len(v)))
        for name in v:
            assert name not in res, "we should never be trying to add a name that's already in the dict"
            res[name] = k

    print("")
    print("BASE_PATH: {}".format(BASE_PATH))
    print("DATA_PATH: {}".format(DATA_PATH))
    print("")

    json_path = os.path.join(BASE_PATH, "fif_split.json")
    print("#3: {}".format(json_path))
    with open(json_path, "w") as of:
        print("Writing to: {}".format(json_path))
        json.dump(res, of)


if __name__ == "__main__": sys.exit(main(sys.argv))