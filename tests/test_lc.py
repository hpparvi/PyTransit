from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from mpi4py.MPI import Wtime
from pytransit.gimenez import Gimenez
from pytransit.orbits_f import orbits as of

ni = 150

## Test the calculation of projected distances
## ===========================================
m1 = Gimenez(npol=200, lerp=False)
m2 = Gimenez(npol=200, lerp=True)

t = np.linspace(0.94,1.06,5000)

t0, p, a, i = 1, 2, 7, 0.5*np.pi

z = of.z_circular(t, t0, p, a, i, 0)
u = [[uu,vv] for uu,vv in zip(np.linspace(0,0.8,20), np.linspace(0,0.4,20))]

pl.plot(m1(z, 0.1, u))
pl.plot(m2(z, 0.1, u))

pl.ylim(0.985, 1.001)
pl.show()
