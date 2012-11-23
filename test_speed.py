import numpy as np
import matplotlib.pyplot as pl

from mpi4py.MPI import Wtime
from gimenez import gimenez as g

z = np.tile(np.linspace(1e-7,1.2,1000),20)

k=0.15
nldc = 2
u = np.array([0.4,0.1])

for npol in [800, 500, 200, 100, 50, 20]:
    tstart = Wtime()
    g.init(npol, nldc)
    for i in range(50):
        g.eval(z, k, u, 0, npol, 2)
    print "npol {:3d}: {:8.6f}".format(npol, Wtime()-tstart)
