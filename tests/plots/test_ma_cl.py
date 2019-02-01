import numpy as np
import matplotlib.pyplot as pl

from pytransit import Gimenez, MandelAgol, MandelAgolCL, z_circular

mgfd = Gimenez()
mgfl = Gimenez(lerp=True)
macl = MandelAgolCL()
mafl = MandelAgol(lerp=True)
mafd = MandelAgol(lerp=False)

u = [[0.1,0.0], [0.2,0.05], [0.5,0.4]]
u1 = [0.5,0.4]

t = np.linspace(-0.2, 0.2, 500)
pl.plot(t, mafd.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), ls='-', c='0.5', lw=1,  alpha=0.5) #, drawstyle='steps-mid')
pl.plot(t, macl.evaluate_t(t, 0.1, u, 0.0, 5, 5, 0.47 * np.pi), 'b-', alpha=0.5) #, drawstyle='steps-mid')
pl.plot(t, mgfd.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi)+1e-4, 'b-',  alpha=0.5) #, drawstyle='steps-mid')
pl.plot(t, mgfl.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi)+2e-4, 'b-',  alpha=0.5) #, drawstyle='steps-mid')

#pl.plot(t, mafl.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), ls='--', c='0.5', lw=3,  alpha=0.5) #, drawstyle='steps-mid')
#pl.plot(t, mafl.evaluate(t, 0.1, u, 0.0, 5, 5, 0.47*np.pi), 'k-',  alpha=0.5) #, drawstyle='steps-mid')
#pl.plot(t, macl.evaluate(t, 0.1, u1, 0.0, 5, 5, 0.47*np.pi), 'b--',  alpha=0.5) #, drawstyle='steps-mid')


pl.ylim(0.98,1.005)
pl.xlim(-0.2,0.2)
pl.show()
