import numpy as np
import matplotlib.pyplot as pl
from pytransit import Gimenez, MandelAgol
from pytransit.orbits_f import orbits as of

t = np.linspace(-0.2, 0.2, 500)
z = np.abs(t)

m_ma = MandelAgol()
m_gm = Gimenez()

#pl.plot(t, m_ma(z, 0.1, [0.3, 0.1], 0.0), 'b-',  alpha=0.5)
#pl.plot(t, m_gm(z, 0.1, [0.3, 0.1], 0.0), 'k--', alpha=1.0)

pl.plot(t, m_ma.evaluate(t, 0.1, [0.3, 0.1], 0.0, 5, 5, 0.47*np.pi), 'b-',  alpha=0.5)
pl.plot(t, m_gm.evaluate(t, 0.1, [0.3, 0.1], 0.0, 5, 5, 0.47*np.pi), 'k--',  alpha=1.0)

pl.show()
