from numpy import *
from numpy.random import normal, uniform

from matplotlib import rc
from matplotlib.pyplot import subplots, setp
rc('figure', figsize=(13,5))

def plot_lc(time, flux, c=None, ylim=(0.9865, 1.0025), ax=None, figsize=None):
    if ax is None:
        fig, ax = subplots(figsize=figsize)
    else:
        fig, ax = None, ax
    ax.plot(time, flux, c=c)
    ax.autoscale(axis='x', tight=True)
    setp(ax, xlabel='Time [d]', ylabel='Flux', xlim=time[[0, -1]], ylim=ylim)

    if fig is not None:
        fig.tight_layout()
    return ax
