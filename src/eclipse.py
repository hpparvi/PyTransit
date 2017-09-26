from .mandelagol import MandelAgol as MA

class Eclipse(object):
    def __init__(self, supersampling=1, exptime=0.020433598):
        self._model = MA(nldc=0, eclipse=True, supersampling=supersampling, exptime=exptime)
    
    def evaluate(self, t, k, t0, p, a, i, e=0, w=0, c=0.):
        return (k**2 + self._model.evaluate(t, k, [], t0, p, a, i, e, w, c) - 1.)/k**2
        
    def __call__(self, z, k, c=0):
        return (k**2 + self._model(z, k, [], c) - 1.)/k**2
