from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from pytransit.gimenez import Gimenez

from pytransit.orbits_cl import Orbit
from pytransit.orbits_f import orbits as of

ni = 150

## Test the calculation of projected distances
## ===========================================
o = Orbit()
t = np.linspace(0,5,500)

t0, p, a, i = 1, 2, 7, 0.49*np.pi

#pl.plot(o.z_circular(t, t0, p, a, i))
pl.plot(t, of.z_circular(t, t0, p, a, i, 0))

pl.show()
exit()


m_cpu = Gimenez()
m_gpu = Gimenez_cl()

z  = np.tile(np.linspace(1e-7,1.2,1000),20)
u  = np.array([0.4,0.1])

for npol in [200, 100, 50, 20]:
    m_cpu = Gimenez(npol=npol, nldc=2, lerp=True)
    m_ocl = GimenezCL(npol=npol, nldc=2, lerp=True)
    tstart = Wtime()
    for i in range(ni):
        m_cpu(z, 0.1, u, update=True if i==0 else False)
    t_cpu = Wtime() - tstart

    tstart = Wtime()
    for i in range(ni):
        m_ocl(z, 0.1, u, update=True if i==0 else False)
    t_ocl = Wtime() - tstart

    #pl.plot(m_cpu(z, 0.1, u))
    #pl.plot(m_ocl(z, 0.1, u))
    #pl.show()
    #exit()
    print "npol {:3d}: {:8.6f} s/lc  -- {:8.6f} s/lc".format(npol, t_cpu/ni, t_ocl/ni)

