import numpy as np
import matplotlib.pyplot as pl
from gimenez import gimenez as g

z = np.linspace(1e-7,1.5,1000)

k=0.15
nldc = 2
u = np.array([0.0,0.0])

g.init(800, nldc)
I_800 = g.eval(z, k, u, 0, 800, 2)

pl.plot(I_800, c='0.0')
pl.show()
