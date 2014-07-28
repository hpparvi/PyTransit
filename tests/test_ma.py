import numpy as np
import matplotlib.pyplot as pl
from pytransit import Gimenez, MandelAgol
from pytransit.mandelagol_f import mandelagol as ma

m_ma = MandelAgol()
m_gm = Gimenez()

ed,le,ld,kt,zt = ma.calculate_interpolation_tables(0.10,0.14,100,256,4)

#pl.plot(zt, ld[:2,:].T); pl.show(); exit()

z = np.abs(np.linspace(-1.3,1.3,100)) + 1e-3
flux = ma.eval_quad_bilerp(z, 0.12, [0.3,0.1], 0.0, 4, ed, le, ld,kt,zt)

print zt
pl.plot(flux)
pl.show()
#pl.imshow(ld);pl.show()

exit()

t = np.linspace(-0.2, 0.2, 500)
pl.plot(t, m_ma.evaluate(t, 0.1, [[0.3, 0.1],[0.4,0.2],[0.6,0.3]], 0.0, 5, 5, 0.47*np.pi), 'b-',  alpha=0.5)
pl.plot(t, m_gm.evaluate(t, 0.1, [[0.3, 0.1],[0.4,0.2],[0.6,0.3]], 0.0, 5, 5, 0.47*np.pi), 'k--', alpha=1.0)
pl.show()
