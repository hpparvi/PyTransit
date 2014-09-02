from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration
import distutils.sysconfig as ds

setup(name='PyTransit',
      version='1.0',
      description='Fast and painless exoplanet transit light curve modelling.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/PyTransit',
      extra_options = ['-fopenmp'],
      package_dir={'pytransit':'src'},
      packages=['pytransit'],
      package_data={'':['*.cl']},
      ext_modules=[Extension('pytransit.gimenez_f', ['src/gimenez.f90'], libraries=['gomp','m'], define_macros=[('DCHUNK_SIZE',128)]),
                   Extension('pytransit.mandelagol_f', ['src/mandelagol.f90'], libraries=['gomp','m']),
                   Extension('pytransit.orbits_f',  ['src/orbits.f90','src/orbits.pyf'], libraries=['gomp','m'])],
      install_requires=["numpy"],
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Programming Language :: Fortran",
          "Programming Language :: OpenCL"
      ]
     )
