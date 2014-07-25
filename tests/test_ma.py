import numpy as np
import matplotlib.pyplot as pl
from pytransit import Gimenez, MandelAgol

m_ma = MandelAgol()
m_gm = Gimenez()

t = np.linspace(-0.2, 0.2, 500)
pl.plot(t, m_ma.evaluate(t, 0.1, [[0.3, 0.1],[0.4,0.2],[0.6,0.3]], 0.0, 5, 5, 0.47*np.pi), 'b-',  alpha=0.5)
pl.plot(t, m_gm.evaluate(t, 0.1, [[0.3, 0.1],[0.4,0.2],[0.6,0.3]], 0.0, 5, 5, 0.47*np.pi), 'k--', alpha=1.0)
pl.show()
