#! /usr/bin/env python
from __future__ import division, print_function
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os, glob
import h5py


if len(sys.argv) <= 1:
    if check_output(["hostname"]).startswith("helios"):
        path = "/home/julesgm/COCO/experimentations/scores"
    else:
        path = os.path.dirname(__file__) + "/scores/"

    print(path)
    os.chdir(path)
    print(os.listdir(os.getcwd()))
    files = glob.glob("*.h5")
    name = sorted(files, key=lambda x: x.split("_")[-1])[-1]
    name = os.path.abspath(name)

else:
    name = os.path.abspath(sys.argv[1])

print("\ntrying to open {}. Does it exist: {}\n".format(name, os.path.exists(name)))

_file = h5py.File(name, "r")

print(_file.keys())

scores_training = _file["scores_training"]
scores_valid = _file["scores_valid"]

plt.plot(np.arange(scores_training.shape[0]), scores_training)
plt.plot(np.arange(scores_valid.shape[0]), scores_valid)
fig = plt.gcf()
fig.canvas.set_window_title("display.py")
plt.show()
