from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from mpi4py.MPI import Wtime
from pytransit.gimenez import Gimenez

ni = 250
z  = np.tile(np.linspace(1e-7,1.2,1000),20)
us = [ [0.2,0.1],
      [[0.2,0.1],[0.4,0.2]],
      [ 0.2,0.1 , 0.4,0.2 , 0.6,0.3],
      [[0.2,0.1],[0.4,0.2],[0.6,0.3], [0.0, 0.4]],
       [[uu,vv] for uu,vv in zip(np.linspace(0,0.8,20), np.linspace(0,0.4,20))]
]

for u in us:
    for npol in [200, 100, 50, 20]:
        tstart = Wtime()
        m = Gimenez(npol=npol)
        for i in range(ni):
            m(z, 0.1, u)
        print "npb {:3d}:  npol {:3d}: {:8.6f} s/lc".format(len(u), npol, (Wtime()-tstart)/(ni))
    print
