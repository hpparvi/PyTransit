import numpy as np
import matplotlib.pyplot as pl
from pytransit import Gimenez, MandelAgol, z_circular

m_mad = MandelAgol()
m_mal = MandelAgol(lerp=True)
m_gmd = Gimenez()

t = np.linspace(0,2,100)
pl.plot(t, z_circular(t, 1, 1, 3, 0.5*np.pi, nthreads=4))
pl.show()

u = [[0.1,0.1],[0.2,0.2],[0.6,0.3]]
#u = [0.2,0.2]

t = np.linspace(-0.2, 0.2, 500)
pl.plot(t, m_mad.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'b-',  alpha=0.5)
pl.plot(t, m_mal.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'r-',  alpha=0.5)
pl.plot(t, m_gmd.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'k--', alpha=1.0)
pl.show()
