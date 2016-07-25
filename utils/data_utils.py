from __future__ import print_function, generators, division
from six import iteritems
import os, sys, re, glob, warnings, logging, enum, json
import mne.io.pick
import mne
import numpy as np
mne.set_log_level("ERROR")


def data_gen(base_path, limit = None):
    """

    The objective is to never have exceptions, to always know what to ignore and why.

    Test script:
        to helios 1>/dev/null; ssh helios 'cd COCO; python -c "from data_utils import data_gen; [x for x in data_gen()]"'

    """

    base_path = os.path.abspath(base_path)
    assert os.path.exists(base_path), "{base_path} doesn't exist".format(base_path=base_path)
    full_glob = glob.glob(base_path + "/*.fif")

    if len(full_glob) == 0:
        raise RuntimeError("Datagen didn't find find any '.fif' files")

    print("glob found {} .fif files".format(len(full_glob)))

    if limit is not None:
        print(">>>>>>>>>> Warning: data_gen limit argument is not None.\n"
              ">>>>>>>>>> This has the effect that only a limited amount ({})\n"
              ">>>>>>>>>> of the data will be loaded. \n".format(limit))

    fif_paths = full_glob[:limit] if limit is not None else full_glob

    if len(fif_paths) == 0:
        raise RuntimeError("fif_path is of size zero.")

    failed = 0
    for fif_path in fif_paths:
        logging.info("Ignored ratio: {}" .format(failed / len(fif_paths)))
        name = fif_path.split("/")[-1] # os.path.split appears broken somehow

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = mne.io.Raw(fif_path)
            except ValueError, err:
                logging.error("-- data_gen ValueError:")
                logging.error("-- %s" % name)
                logging.error("-- %s\n" % err)
                raise err
            except TypeError, err:
                logging.error("-- data_gen TypeError")
                logging.error("-- %s" % name)
                logging.error("-- %s\n" % err)
                raise err

        assert name.lower().startswith("k") or name.lower().startswith("r"), "file name is weird, can't guess label from it. ({})".format(name)
        label = name.lower().startswith("k")

        yield name, raw, label

