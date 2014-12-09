import numpy as np
import matplotlib.pyplot as pl
from pytransit.gimenez import Gimenez

k, t0, p, a, i, e, w = 0.1, 1., 4., 8., 0.5*np.pi, 0.01, 0.5*np.pi
t = np.linspace(0.9,1.1,500)
u = np.array([[0.04*iu, 0.025*iu, 0.01*iu] for iu in range(10)])
c = np.linspace(0,0.9,10)

m = Gimenez(lerp=False, npol=100, nldc=3)
f = m.evaluate(t, k, u, t0, p, a, i, e, w, c=c)

fig,ax = pl.subplots(1,1,figsize=(7,7))
ax.plot(t,f + np.arange(u.shape[0])*0.00, 'k');
pl.setp(ax, ylim=(0.9875,1.011), yticks=[], xlabel='Time [d]', ylabel='Normalised flux', 
        title='Light curve for a set of multiple limb darkening coefficients');
fig.tight_layout()
pl.show()
