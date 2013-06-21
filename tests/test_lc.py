from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from mpi4py.MPI import Wtime
from pytransit.gimenez import Gimenez
from pytransit.orbits_f import orbits as of

ni = 150

## Test the calculation of projected distances
## ===========================================
m = Gimenez()
t = np.linspace(0.94,1.06,5000)

t0, p, a, i = 1, 2, 7, 0.49*np.pi

z = of.z_circular(t, t0, p, a, i, 0)
u = np.array([[0.2,0.1],[0.4,0.2],[0.6,0.3]]).T


lc = m(z, 0.1, u)

pl.plot(lc)
pl.ylim(0.985, 1.001)
pl.show()
