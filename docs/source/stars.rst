Stellar spectra
===============

``pytransit.stars`` gives access to two grids of theoretical stellar spectra that PyTransit ships
with. They are what the :doc:`contamination` models integrate over passbands, and what the
:doc:`gravity-darkened model <models/gdmodel>` uses to convert local surface temperature into
observed flux.

The grids
---------

Both grids are *precomputed and binned*. The original spectra are averaged over surface gravity and
metallicity for each effective temperature and binned to a uniform 1 nm wavelength grid. This keeps
them small enough to distribute inside the package -- together about 18 MB -- while remaining
accurate enough for the broadband flux ratios they are used for.

.. list-table::
    :header-rows: 1
    :widths: 22 22 22 34

    * - Grid
      - Wavelengths
      - Temperatures
      - Reference
    * - BT-Settl
      - 10-30000 nm
      - 1200-7000 K, 68 steps
      - Allard et al. (2012)
    * - Husser2013
      - 300-2500 nm
      - 2300-12000 K, 73 steps
      - Husser et al. (A&A 553, A6, 2013)

**BT-Settl** covers a much wider wavelength range and reaches down to brown dwarf temperatures,
which makes it the grid of choice for cool hosts, M dwarfs, and low-mass contaminants. It is the
default in :class:`~pytransit.contamination.contamination.SMContamination`.

**Husser2013** extends to hotter stars, up to early A types, but over a narrower wavelength range.

Neither grid resolves individual lines after binning. They are meant for integrating over
passbands, not for spectroscopy.

Reading a grid
--------------

Each grid has a ``read_*_table`` function returning a `pandas.DataFrame` with the effective
temperatures as the index and the wavelengths as the columns.

.. code-block:: python

    from pytransit.stars import read_bt_settl_table

    df = read_bt_settl_table()
    df.shape
    # (68, 29991)

    df.index.values[:5]        # effective temperatures in K
    df.columns.values[:5]      # wavelengths in nm

Interpolating
-------------

For a spectrum at an arbitrary temperature, ``create_*_interpolator`` returns a
`scipy.interpolate.RegularGridInterpolator` over (temperature, wavelength).

.. code-block:: python

    from numpy import linspace, full, transpose
    from pytransit.stars import create_bt_settl_interpolator

    ip = create_bt_settl_interpolator()

    wl = linspace(400.0, 900.0, 500)
    spectrum = ip(transpose([full(wl.size, 5500.0), wl]))

The interpolator is built from the full table each time it is created, so create it once and reuse
it.

.. note::

    The interpolators do not extrapolate. Requesting a temperature or wavelength outside the grid
    raises a `ValueError`.

Rebuilding the tables
---------------------

``compute_averaged_bt_settl_table`` and ``compute_averaged_husser2013_table`` regenerate the shipped
tables from a locally downloaded spectrum grid. These are package maintenance functions -- PyTransit
comes with the tables precomputed, so they only need to be run to rebuild them.

API
---

BT-Settl
********

.. autofunction:: pytransit.stars.read_bt_settl_table

.. autofunction:: pytransit.stars.create_bt_settl_interpolator

.. autofunction:: pytransit.stars.compute_averaged_bt_settl_table

.. py:data:: pytransit.stars.bt_settl_file

    Absolute path to the FITS file holding the binned BT-Settl table.

Husser et al. (2013)
********************

.. autofunction:: pytransit.stars.read_husser2013_table

.. autofunction:: pytransit.stars.create_husser2013_interpolator

.. autofunction:: pytransit.stars.compute_averaged_husser2013_table

.. py:data:: pytransit.stars.husser2013_file

    Absolute path to the FITS file holding the binned Husser et al. (2013) table.
