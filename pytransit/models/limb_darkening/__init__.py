"""Numba-accelerated limb darkening models.

Each model is defined by an intensity profile function ``ld_<model>(mu, pv)``
and, where an analytic solution exists, a disk-integrated stellar intensity
function ``ldi_<model>(pv)`` returning 2*pi * int_0^1 I(mu) mu dmu, and a
derivative function ``ldd_<model>(mu, pv)``.
"""
from .evaluation import evaluate_ld, evaluate_ldi
from .exponential import ld_exponential, ldi_exponential
from .general import ld_general, ldi_general
from .linear import ld_linear, ldi_linear, ldd_linear
from .logarithmic import ld_logarithmic, ldi_logarithmic
from .nonlinear import ld_nonlinear, ldi_nonlinear
from .power_2 import ld_power_2, ldi_power_2, ldd_power_2, ld_power_2_pm, ldi_power_2_pm
from .quadratic import ld_quadratic, ldi_quadratic, ldd_quadratic
from .quadratic_tri import ld_quadratic_tri, ldi_quadratic_tri
from .square_root import ld_square_root, ldi_square_root
from .uniform import ld_uniform, ldi_uniform

__all__ = ['ld_uniform', 'ldi_uniform',
           'ld_linear', 'ldi_linear', 'ldd_linear',
           'ld_quadratic', 'ldi_quadratic', 'ldd_quadratic',
           'ld_quadratic_tri', 'ldi_quadratic_tri',
           'ld_nonlinear', 'ldi_nonlinear',
           'ld_general', 'ldi_general',
           'ld_power_2', 'ldi_power_2', 'ldd_power_2',
           'ld_power_2_pm', 'ldi_power_2_pm',
           'ld_square_root', 'ldi_square_root',
           'ld_logarithmic', 'ldi_logarithmic',
           'ld_exponential', 'ldi_exponential',
           'evaluate_ld', 'evaluate_ldi']
