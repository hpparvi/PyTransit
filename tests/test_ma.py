import numpy as np
import matplotlib.pyplot as pl
from pytransit import ma_quad_f

t = np.linspace(-1.2, 1.2, 500)
z = np.abs(t)

pl.plot(t, ma_quad_f.ma_quad.ma(z, 0.1, 0.3, 0.1))
pl.show()
