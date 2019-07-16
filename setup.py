from setuptools import setup, find_packages

version = '2.0'

setup(name='PyTransit',
      version=version,
      description='Fast and painless exoplanet transit light curve modelling.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/PyTransit',
      package_dir={'pytransit':'src'},
      packages=['pytransit', 'pytransit.models', 'pytransit.models.numba', 'pytransit.models.opencl', 'pytransit.orbits',
                'pytransit.utils', 'pytransit.param', 'pytransit.contamination','pytransit.lpf', 'pytransit.lpf.baselines'],
      package_data={'':['*.cl'], 'pytransit.contamination':['data/*']},
      install_requires=["numpy", "numba", "scipy", "pandas", "xarray", "tables"],
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
          "Programming Language :: Fortran",
          "Programming Language :: Other"
      ],
      keywords='astronomy astrophysics exoplanets'
      )
