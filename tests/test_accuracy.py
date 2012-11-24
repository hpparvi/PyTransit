import numpy as np
import matplotlib.pyplot as pl
from gimenez import gimenez as g

z = np.linspace(1e-7,1.5,1000)

k=0.15
nldc = 2
u = np.array([0.0,0.0])

g.init(800, nldc)
I_800 = g.eval(z, k, u, 0, 800, 2)

g.init(500, nldc)
I_500 = g.eval(z, k, u, 0, 500, 2)

g.init(200, nldc)
I_200 = g.eval(z, k, u, 0, 200, 2)

g.init(100, nldc)
I_100 = g.eval(z, k, u, 0, 100, 2)

g.init(50, nldc)
I_50 = g.eval(z, k, u, 0, 50, 2)

g.init(20, nldc)
I_20 = g.eval(z, k, u, 0, 20, 2)

print 1e3*np.abs(I_800-I_20).max()

pl.plot(I_800, c='0.0')
pl.plot(I_50, c='0.5')
pl.show()
