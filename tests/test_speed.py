from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from mpi4py.MPI import Wtime
from pytransit.gimenez import Gimenez

ni = 150

z  = np.tile(np.linspace(1e-7,1.2,1000),20)
u  = np.array([[0.2,0.1],[0.4,0.2],[0.6,0.3]]).T

for npol in [200, 100, 50, 20]:
    tstart = Wtime()
    m = Gimenez(npol=npol)
    for i in range(ni):
        m(z, 0.1, u)
    print "npol {:3d}: {:8.6f} s/lc".format(npol, (Wtime()-tstart)/ni)
