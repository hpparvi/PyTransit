import numpy as np
import matplotlib.pyplot as pl
from pytransit import Gimenez, MandelAgol, z_circular

m_mad = MandelAgol()
m_mal = MandelAgol(interpolate=True)
m_gmd = Gimenez(interpolate=False)
m_gml = Gimenez(interpolate=True)

u = np.array([[0.1,0.1],[0.2,0.2],[0.6,0.3]])

t = np.linspace(-0.2, 0.2, 500)
pl.plot(t, m_mad.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'b-',  alpha=0.5)
pl.plot(t, m_mal.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'r-',  alpha=0.5)
pl.plot(t, m_gmd.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'k--', alpha=1.0)
pl.show()
