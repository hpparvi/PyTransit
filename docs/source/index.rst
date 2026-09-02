.. title:: PyTransit docs

PyTransit
=========

PyTransit is a package for exoplanet transit light curve modelling. It offers optimised CPU and
GPU implementations of exoplanet transit models behind a unified interface. Evaluating a model is
trivial for simple cases, such as a homogeneous light curve observed in a single passband, and
stays straightforward for complex ones, such as *heterogeneous light curves containing transits
observed in different passbands with different instruments*, or *transmission spectroscopy*.

Development began in 2009 to fill the need for a fast and reliable exoplanet transit modelling
toolkit for Python. PyTransit has since gone through several iterations, always aiming to be *the
fastest and most versatile* exoplanet transit modelling tool for Python.

Features
--------

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: Highlights
        :link: highlights
        :link-type: doc

        Light curve, passband and epoch indices: variable supersampling, multicolour photometry
        and TTVs, straight out of the box.

    .. grid-item-card:: RoadRunner model
        :link: features/roadrunner
        :link-type: doc

        A fast transit model that works with *any* radially symmetric limb darkening profile.

    .. grid-item-card:: Transmission spectroscopy
        :link: features/tsmodel
        :link-type: doc

        Thousands of wavelength bins sharing one transit geometry, computed once.

    .. grid-item-card:: Oblate planet
        :link: features/opmodel
        :link-type: doc

        Tens-of-ppm oblateness signals, with the model accuracy dialled to match.

    .. grid-item-card:: Gravity-darkened star
        :link: features/gdmodel
        :link-type: doc

        Transits across a flattened, gravity-darkened fast rotator, and the spin-orbit angle
        their asymmetry reveals.

    .. grid-item-card:: Chromosphere
        :link: features/chromosphere
        :link-type: doc

        Transits over a limb-*brightened* optically thin shell, which inverts the usual shape.

.. toctree::
    :hidden:

    features/index

Documentation
-------------

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Getting started
        :link: installation
        :link-type: doc

        Install PyTransit and evaluate your first transit model.

    .. grid-item-card:: User guide
        :link: guide/index
        :link-type: doc

        The model interface, data setup, evaluation, limb darkening, and the OpenCL backend.

    .. grid-item-card:: Transit models
        :link: models/index
        :link-type: doc

        The model catalogue: what each model does, when to use it, and its full API.

    .. grid-item-card:: API reference
        :link: api/index
        :link-type: doc

        The transit model base class, limb darkening laws, stellar spectra, contamination, and I/O.

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Getting started

    installation
    highlights
    notebooks/quickstart

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: User guide

    guide/index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Transit models

    models/index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Supporting modules

    stars
    contamination
    io

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Reference

    api/index

Citing PyTransit
----------------

If PyTransit contributes to a publication, please cite Parviainen (MNRAS 450, 3233, 2015). The
individual models carry their own references, listed on each model's page under
:doc:`models/index`.

Support
-------

If you run into difficulties with PyTransit, please open an issue on the
`GitHub repository <https://github.com/hpparvi/PyTransit/issues>`_. Suggestions and feature
requests are welcome through the same route.

License
-------

PyTransit is licensed under the `GPLv3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_ license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
