import numpy as np
import matplotlib.pyplot as pl
from pytransit import ma_quad_f
from pytransit.mandelagol import MandelAgol
from pytransit.gimenez import Gimenez

t = np.linspace(-1.2, 1.2, 500)
z = np.abs(t)

m_ma = MandelAgol()
m_gm = Gimenez()

pl.plot(t, m_ma(z, 0.1, [0.3, 0.1], 0.0), 'b-',  alpha=0.5)
pl.plot(t, m_gm(z, 0.1, [0.3, 0.1], 0.0), 'k--', alpha=1.0)
pl.show()
