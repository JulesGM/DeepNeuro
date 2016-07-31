#! /usr/bin/env python
from __future__ import print_function, division, with_statement
import sys, os, subprocess as sp, glob, re

BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
REMOTE_PATH = '/home/julesgm/COCO'

assert BASE_PATH == "/home/jules/Documents/COCO_NEURO"

d_target = "helios"

if len(sys.argv) > 1:
    target = sys.argv[1]

else:
    print("Defaulting to {}".format(d_target))
    target = d_target

exclude_list = [
                "modules_repo/",
                "scores/",
                "fif_split.json",
                ".git/",
                "*.json",
                "latest_save.h5",
                "*.pkl",
                ]

abs_exclude_set = {re.sub("/+", "/", "{}/{}".format(BASE_PATH, rel)) for rel in exclude_list}
abs_exclude_set.update({re.sub("/+", "/", "{}".format(rel)) for rel in glob.glob(BASE_PATH + "/" + "*.pyc")})

include_set = set([BASE_PATH + "/"])

# Doing everything we can to prevent having to do "shell=True"
cmd = (["rsync",
        "--partial",
        "--progress",
        "-r",
        ]
          + list(include_set)
          + ["{target}:\"{remote_path}/\"".format(target=target, remote_path=REMOTE_PATH)])

for exclude_unit in exclude_list:
    cmd += ["--exclude", "{exclude_unit}".format(base_path=BASE_PATH,
                exclude_unit=exclude_unit)]

print(" ".join(cmd))

sp.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout).wait()
