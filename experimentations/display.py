#! /usr/bin/env python

from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys, os, glob

base = "/home/julesgm/COCO/experimentations/"

if len(sys.argv) > 1:
	target = os.path.abspath(sys.argv[1])
else:
	target = next(reversed(sorted(glob.glob(os.path.join(base, "*.out")))))

cmd = r"grep 'score:' '%s' | sed 's/score:\s\{2,\}//' | grep -v ':\s0' | grep -v 'oob'" % os.path.abspath(target)
print(cmd)

res = check_output(cmd, shell=True)
reres = np.array([float(x) for x in res.split()])
plt.plot(np.arange(len(reres)), reres)
plt.title(target.split("/")[-1])
fig = plt.gcf()
fig.canvas.set_window_title("display.py")
plt.show()
