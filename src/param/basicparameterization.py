from math import sqrt, acos, atan2
from numpy import array, inf, pi

from pytransit.utils.orbits import as_from_rhop, i_from_baew
from pytransit.param.parameter import Parameter

class Parameterization(object):
    pars = []

    def __new__(cls):
        cls.__initialize_params__()
        return super(Parameterization, cls).__new__(cls)
    
    @classmethod
    def __initialize_params__(cls):
        cls.names    = [p.name for p in cls.pars]
        cls.abbrevs  = [p.abbr for p in cls.pars]
        cls._descrs  = [p.desc for p in cls.pars]
        cls._lims    = [p.lims for p in cls.pars]
        cls._defvs   = [p.defv for p in cls.pars]
        
        for i, (name, abbr, desc) in enumerate(zip(cls.names, cls.abbrevs, cls._descrs)):
            exec('def _get_{:s}(self): return self._data[{:d}]'.format(name, i))
            exec('def _set_{:s}(self, v): self._data[{:d}] = v'.format(name, i))
            exec('cls.{nm:s} = cls.{ab:s} = property(_get_{nm:s}, _set_{nm:s}, doc="{des:s}")'.format(ab=abbr, des=desc, nm=name))
        
        
    def __init__(self):
        self._data = array(self._defvs)
        
    def __len__(self):
        return self._data.size
    
    def __getitem__(self, key):
        return self._data.__getitem__(key)
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def __repr__(self):
        return '\n'.join([p for p in self.pars])


class ModelParameterization(Parameterization):
    
    pars = (Parameter('zero_epoch',       'tc',  'Zero epoch',             (-inf,  inf), 0),       # 0
            Parameter('period',           'p',   'Orbital period',         (   0,  inf), 4),       # 1
            Parameter('radius_ratio',     'k',   'Radius ratio',           (   0,  inf), 0.1),     # 2
            Parameter('semi_major_axis',  'sma', 'Semi-major axis',        (   0,  inf), 2),       # 3
            Parameter('inclination',      'i',   'Inclination',            (   0,   pi), 0.5*pi),  # 4
            Parameter('eccentricity',     'e',   'Eccentricity',           (   0,    1), 0.0),     # 5
            Parameter('omega',            'w',   'Argument of periastron', (   0, 2*pi), 0.0),     # 6
            Parameter('ld_u',             'u',   'Limb darkening u',       (   0,    1), 0),       # 7
            Parameter('ld_v',             'v',   'Limb darkening v',       (   0,    1), 0))       # 8
            
    def __new__(cls):
        cls.__initialize_params__()
        return super(ModelParameterization, cls).__new__(cls)
    
    def to_tmodel(self, pv=None):
        if pv is not None:
            self._data[:] = pv
        pv = self._data
        return pv[2], pv[7:9], pv[0], pv[1], pv[3], pv[4], pv[5], pv[6]

    
class BasicCircularParameterization(Parameterization):
    
    pars = (Parameter('zero_epoch',       'tc',  'Zero epoch',        (-inf, inf), 0),     # 0
            Parameter('period',           'p',   'Orbital period',    (   0, inf), 4),     # 1
            Parameter('area_ratio',       'ar',  'Area ratio',        (   0, inf), 0.01),  # 2
            Parameter('stellar_density',  'rho', 'Stellar density',   (   0, inf), 2),     # 3
            Parameter('impact_parameter', 'b',   'Impact parameter',  (   0,   1), 0),     # 4
            Parameter('ld_q1',            'q1',  'Limb darkening q1', (   0,   1), 0),     # 5
            Parameter('ld_q2',            'q2',  'Limb darkening q2', (   0,   1), 0))     # 6
        
    radius_ratio     = k   = property(lambda self: sqrt(self._data[2]), doc='Radius ratio')
    semi_major_axis  = sma = property(lambda self: as_from_rhop(self._data[3], self._data[1]), doc='Semi-major axis')
    inclination      = i   = property(lambda self: acos(self._data[4]/self.sma), doc='Inclination')
    
    def __new__(cls):
        cls.__initialize_params__()
        return super(BasicCircularParameterization, cls).__new__(cls)
    
    def to_tmodel(self, pv=None):
        if pv is not None:
            self._data[:] = pv
        pv = self._data
        
        sma  = as_from_rhop(pv[3], pv[1])   # Scaled semi-major axis from stellar density and orbital period
        inc  = acos(pv[4]/sma)              # Inclination from impact parameter and semi-major axis
        k    = sqrt(pv[2])                  # Radius ratio from area ratio
        a,b = sqrt(pv[5]), 2*pv[6]          # Mapping the limb darkening coefficients from the Kipping (2014) 
        uv = array([a*b, a*(1.-b)])         # parameterisation to quadratic MA coefficients
        return k, uv, pv[0], pv[1], sma, inc
    


class BasicEccentricParameterization(BasicCircularParameterization):

    eccentricity = e  = property(lambda self: self._data[7]**2 + self._data[8]**2, doc='Eccentricity')
    omega        = w  = property(lambda self: atan2(self._data[8], self._data[7]), doc='Argument of perioastron')
    inclination  = i  = property(lambda self: i_from_baew(self._data[4], self.sma, self.e, self.w), doc='Inclination')

    def __new__(cls):
        o = super(BasicEccentricParameterization, cls).__new__(cls)
        cls.pars += (Parameter('secw', 'secw', 'sqrt(e) cos(w)', (-1,1), 0),  # 7
                     Parameter('sesw', 'sesw', 'sqrt(e) sin(w)', (-1,1), 0))  # 8
        cls.__initialize_params__()
        return o

    
    def __init__(self):
        super(BasicEccentricParameterization, self).__init__()
        self._data = array([0, 4, 0.01, 2, 0, 0, 0, 0, 0])

        
    def to_tmodel(self, pv=None):
        if pv is not None:
            self._data[:] = pv
        pv = self._data

        sma  = as_from_rhop(pv[3], pv[1])    # Scaled semi-major axis from stellar density and orbital period
        k    = sqrt(pv[2])                   # Radius ratio from area ratio
        a,b  = sqrt(pv[5]), 2*pv[6]          # Mapping the limb darkening coefficients from the Kipping (2014) 
        uv   = array([a*b, a*(1.-b)])        # parameterisation to quadratic MA coefficients
        e    = pv[7]**2 + pv[8]**2           # Eccentricity
        w    = atan2(pv[8], pv[7])           # Argument of periastron
        inc  = i_from_baew(pv[4], sma, e, w) # Inclination from impact parameter, semi-major axis, eccentricity, and argument of periastron
        
        return k, uv, pv[0], pv[1], sma, inc, e, w 
