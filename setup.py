import sys
from numpy.distutils.core import setup, Extension

e_gimenez = Extension('pytransit.gimenez_f', ['src/gimenez.f90'], libraries=['gomp', 'm'], define_macros=[('DCHUNK_SIZE', 128)])
e_mandelagol = Extension('pytransit.mandelagol_f', ['src/mandelagol.f90'], libraries=['gomp', 'm'])
e_utils = Extension('pytransit.utils_f', ['src/utils.f90'], libraries=['gomp', 'm'])
e_orbits = Extension('pytransit.orbits_f',  ['src/orbits.f90','src/orbits.pyf'], libraries=['gomp','m'])

version = '1.5'

setup(name='PyTransit',
      version=version,
      description='Fast and painless exoplanet transit light curve modelling.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/PyTransit',
      extra_options = ['-fopenmp'],
      package_dir={'pytransit':'src'},
      packages=['pytransit','pytransit.utils','pytransit.param', 'pytransit.contamination'],
      package_data={'':['*.cl'], 'pytransit.contamination':['data/*']},
      ext_modules=[e_gimenez, e_mandelagol, e_utils, e_orbits],
      install_requires=["numpy", "scipy", "pandas", "xarray", "tables"],
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
          "Programming Language :: Other"
      ]
      )
