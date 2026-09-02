Installation
============

PyTransit can be installed from PyPI

.. code-block:: bash

    pip install pytransit

using conda

.. code-block:: bash

    conda install conda-forge::pytransit

or from source

.. code-block:: bash

    git clone https://github.com/hpparvi/PyTransit.git
    cd PyTransit
    pip install .

Add ``-e`` to the last command for an editable install if you intend to modify the code.

Requirements
------------

PyTransit needs Python 3 with NumPy, SciPy, pandas, Numba, Astropy, and Matplotlib. These are all
installed automatically.

Optional dependencies
---------------------

Two features depend on packages that are not installed automatically.

**PyOpenCL** enables the GPU models described in :doc:`guide/opencl`. It also needs a working
OpenCL runtime for your device, which is installed separately and depends on your hardware and
operating system.

.. code-block:: bash

    pip install pyopencl

**LDTk** enables :class:`~pytransit.models.ldtkldm.LDTkLDModel`, which computes limb darkening
profiles from stellar atmosphere models rather than from an analytic law. See
:doc:`guide/limb_darkening`.

.. code-block:: bash

    pip install ldtk

Both imports are guarded, so PyTransit works without them; only the corresponding features are
unavailable.

Verifying the installation
--------------------------

.. code-block:: python

    from numpy import pi, linspace
    from pytransit import RoadRunnerModel

    tm = RoadRunnerModel('quadratic')
    tm.set_data(linspace(-0.1, 0.1, 1000))
    flux = tm.evaluate(k=0.1, ldc=[0.2, 0.1], t0=0.0, p=1.0, a=3.0, i=0.5*pi)

    print(flux.min())     # ~0.989

The first evaluation triggers Numba's just-in-time compilation and takes a few seconds. Subsequent
evaluations are fast.
