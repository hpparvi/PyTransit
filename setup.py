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

from setuptools import setup, find_packages

from version import version

setup(name='PyTransit',
      version=str(version),
      description='Fast and painless exoplanet transit light curve modelling.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/PyTransit',
      package_dir={'pytransit':'pytransit'},
      packages=['pytransit', 'pytransit.models', 'pytransit.models.numba', 'pytransit.models.opencl', 'pytransit.orbits',
                'pytransit.utils', 'pytransit.param', 'pytransit.contamination','pytransit.lpf', 'pytransit.lpf.tess',
                'pytransit.lpf.baselines'],
      package_data={'':['*.cl'], 'pytransit.contamination':['data/*']},
      install_requires=["numpy", "numba", "scipy", "pandas", "xarray", "tables", "semantic_version"],
      include_package_data=True,
      license='GPLv2',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 5 - Production/Stable",
          "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
      ],
      keywords='astronomy astrophysics exoplanets'
      )
