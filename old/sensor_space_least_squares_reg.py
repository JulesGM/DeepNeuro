"""
 http://martinos.org/mne/dev/auto_examples/stats/plot_sensor_regression.html#sphx-glr-auto-examples-stats-plot-sensor-regression-py
 Authors: Tal Linzen <linzen@nyu.edu>
          Denis A. Engemann <denis.engemann@gmail.com>
 License: BSD (3-clause)
 Modified by Jules Gagnon-Marchand <jgagnonmarchand@gmail.com>

"""
import numpy as np

import mne

from mne.stats.regression import linear_regression

print(__doc__)

def plot_topomap(x, unit):
    x.plot_topomap(ch_type='mag', scale=1, size=1.5, vmax=np.max,
                   unit=unit, times=np.linspace(0.1, 0.2, 5))

def sensor_regression(raw, events):
    tmin, tmax = -0.2, 0.5
    event_id = dict(aud_l=1, aud_r=2)


    picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False,
                           eog=False, exclude='bads')

    # Reject some epochs based on amplitude
    reject = dict(mag=5e-12)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=True,
                        reject=reject)
    names = ['intercept', 'trial-count']

    intercept = np.ones((len(epochs),), dtype=np.float)
    design_matrix = np.column_stack([intercept,  # intercept
                                     np.linspace(0, 1, len(intercept))])

    # also accepts source estimates
    lm = linear_regression(epochs, design_matrix, names)




    trial_count = lm['trial-count']

    plot_topomap(trial_count.beta, unit='z (beta)')
    plot_topomap(trial_count.t_val, unit='t')
    plot_topomap(trial_count.mlog10_p_val, unit='-log10 p')
    plot_topomap(trial_count.stderr, unit='z (error)')

