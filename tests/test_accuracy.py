import numpy as np
import matplotlib.pyplot as pl
from gimenez import Gimenez

z = np.linspace(1e-7,1.5,1000)
u = np.array([0.3, 0.1])
k = 0.10

npols = [800, 500, 200, 100, 50, 20, 10, 5]
I = [Gimenez(npol, 2)(z,k,u) for npol in npols]

fig, ax = pl.subplots(1,1)
la = 1-np.linspace(0,0.75,len(npols))
ax.plot(I[0], c='0.0')

for i in range(1,len(npols)):
    print '\tN {:3d} -- Max deviation {:6.4f} ppm'.format(npols[i], 1e3*np.abs(I[0]-I[i]).max())
    ax.plot(I[i], c='0', alpha=la[i])

pl.show()
