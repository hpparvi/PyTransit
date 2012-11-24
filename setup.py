from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

import sys
sys.argv.extend(['config_fc','--fcompiler=gnu95', '--opt="-Ofast -ffast-math"', '--f90flags="-cpp -fopenmp -march=native"'])

conf = Configuration()

setup(name='PyTransit',
      version='0.5',
      description='Tools for exoplanet transit light curve analysis.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='',
      extra_options = ['-fopenmp'],
      package_dir={'pytransit':'src'},
      packages=['pytransit'],
      ext_modules=[Extension('pytransit.gimenez_f', ['src/gimenez.f90'], libraries=['gomp','m'], define_macros=[('DCHUNK_SIZE',128)], extra_f90_compile_args=['-cpp'])]
     )
