#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from numpy import asarray, zeros
import math as mt

class LDParameterization(object):
    __slots__ = ('coefs',)
    name  = ''
    ncoef = 0

    def __init__(self, coefs=None):
        self.coefs = zeros(self.ncoef) if coefs is None else asarray(coefs)
        assert self.coefs.size == self.ncoef
    
    def __str__(self):
        return '{} : {}'.format(self.name, self.coefs)
    
    def __repr__(self):
        return '{} : {}'.format(self.name, self.coefs)

    def map_from(self, other):
        raise NotImplementedError

    @property
    def uniform(self):
        raise NotImplementedError
    
    @property
    def linear(self):
        raise NotImplementedError
        
    @property
    def quadratic(self):
        raise NotImplementedError
        
    @property    
    def triangular(self):
        raise NotImplementedError
        
       
class UniformLD(LDParameterization):
    """Uniform limb darkening parameterization
    """
    __slots__ = ('coefs',)
    name  = 'uniform'
    ncoef = 0
    
    @property
    def uniform(self):
        return self
    
    @property    
    def linear(self):
        return LinearLD([0.])
    
    @property    
    def quadratic(self):
        return QuadraticLD([0., 0.])
    
    @property
    def triangular(self):
        return TriangularQLD([0., 0.])
    
    
class LinearLD(LDParameterization):
    """Linear limb darkening parameterization
    """
    __slots__ = ('coefs')
    name = 'linear'
    ncoef = 1

    def map_from(self, other):
        self.coefs[:] = other.linear.coefs
    
    @property
    def linear(self):
        return self
    
    @property    
    def quadratic(self):
        return QuadraticLD([self.coefs[0], 0])
    
    @property
    def triangular(self):
        return self.quadratic.triangular
    
    
class QuadraticLD(LDParameterization):
    """Quadratic limb darkening parameterization
    """
    __slots__ = ('coefs',)
    name = 'quadratic'
    ncoef = 2

    def map_from(self, other):
        self.coefs[:] = other.quadratic.coefs
    
    @property    
    def quadratic(self):
        return self
    
    @property      
    def triangular(self):
        u,v = self.coefs
        return TriangularQLD([(u+v)**2, u/(2*(u+v))])
        
        
class TriangularQLD(QuadraticLD):
    """Triangular quadratic parametrization by Kipping (2013)
    
    Kipping, D. MNRAS, 435 (3) pp. 2152-2160, 2013
    """
    __slots__ = ('coefs',)
    name = 'triangular quadratic'

    def map_from(self, other):
        self.coefs[:] = other.triangular.coefs
    
    @property
    def quadratic(self):    
        a,b = mt.sqrt(self.coefs[0]), 2*self.coefs[1]
        return QuadraticLD([a*b, a*(1-b)])
    
    @property
    def triangular(self):
        return self
