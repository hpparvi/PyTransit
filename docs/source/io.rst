Data containers
===============

``pytransit.utils.io`` provides containers that hold observations together with their metadata.
They exist because real datasets are not a single array: a light curve comes with a passband, an
instrument, an exposure time, a sector, and a supersampling rate, and keeping those in parallel
lists next to the data is how bookkeeping errors happen.

There are two pairs, one for photometry and one for radial velocities:

.. list-table::
    :header-rows: 1
    :widths: 24 76

    * - Class
      - Holds
    * - :class:`~pytransit.utils.io.lcdata.LCData`
      - One light curve: times, fluxes, errors, covariates, and metadata.
    * - :class:`~pytransit.utils.io.lcdata.LCDataGroup`
      - An ordered collection of `LCData` objects.
    * - :class:`~pytransit.utils.io.rvdata.RVData`
      - One radial velocity dataset: times, velocities, errors, and covariates.
    * - :class:`~pytransit.utils.io.rvdata.RVDataGroup`
      - An ordered collection of `RVData` objects.

.. code-block:: python

    from pytransit.utils.io import LCData, LCDataGroup, RVData, RVDataGroup

A single light curve
--------------------

.. code-block:: python

    from pytransit.utils.io import LCData

    lc = LCData(time, flux, error=error,
                passband='TESS', instrument='TESS',
                sector=14, exptime=0.0014, nsamples=1)

Only `time` and `flux` are required. Everything else has a default, and the white noise estimate
defaults to the point-to-point estimate :math:`\mathrm{std}(\Delta F)/\sqrt{2}`.

.. note::

    `exptime` is in **days**, consistent with the transit models.

    The arrays are stored without copying when they already are ``float64`` ndarrays, so the caller
    and the `LCData` object may share memory.

`LCData` also carries a small toolkit for the routine cleaning steps:

.. code-block:: python

    lc.running_median(width=15)             # running median of the flux
    lc.outlier_mask(nsigma=3.0, width=15)   # which points a clip would remove
    lc.remove_outliers(nsigma=3.0)          # clip them, in place; returns the count
    lc.add_time_covariates(order=2)         # add normalised time and its powers as covariates
    lc.linear_model()                       # least-squares fit of the flux to the covariates

Outlier clipping is done against a running median rather than a global mean, so it removes points
that deviate from the *local* trend and leaves the transit itself alone.

Groups
------

A group is an ordered collection. **The insertion order is the light curve order**, which is what
defines the light curve indices a model will use, so nothing is ever sorted implicitly.

Groups are built by addition or from a sequence:

.. code-block:: python

    lcs = lc1 + lc2 + lc3
    lcs = LCDataGroup([lc1, lc2, lc3])
    lcs = sum(light_curves)                 # a list of LCData objects

Nested groups are flattened, and adding the same `LCData` instance twice raises a `ValueError`.

A group exposes every per-light-curve quantity as a bulk property, which is what makes it easy to
feed a transit model:

.. code-block:: python

    lcs.times          # list of time arrays
    lcs.fluxes         # list of flux arrays
    lcs.errors         # list of error arrays
    lcs.pbids          # passband index per light curve
    lcs.exptimes       # exposure time per light curve
    lcs.nsamples       # supersampling rate per light curve
    lcs.passband_names # the unique passband names, in pbid order
    lcs.lcslices       # slices splitting a concatenated array back into light curves

These map directly onto the arguments of `set_data`, described in :doc:`guide/data_setup`.

Selecting and sorting
---------------------

``select`` returns a new group of the datasets matching all the given criteria. Tuple-valued
attributes such as `passband` and `pids` match by membership, everything else by equality, and a
criterion may be a single value or a sequence of accepted values.

.. code-block:: python

    lcs.select(passband='TESS')
    lcs.select(instrument='MuSCAT2', sector=[14, 15])
    lcs.select(pids=0)                      # light curves with planet 0 transiting

``sorted_by`` returns a new group in a different order, keyed by ``'time'`` (the default), by any
attribute name, or by a callable.

.. code-block:: python

    lcs.sorted_by()                 # by first observation time
    lcs.sorted_by('passband')
    lcs.sorted_by(lambda d: d.npt)

Groups also support indexing, slicing, boolean masks, and iteration.

.. important::

    ``select``, ``sorted_by``, and slicing return new *groups*, but the `LCData` objects inside them
    are shared by reference, not copied. Modifying a light curve through one group modifies it in
    every group holding it.

Inspecting and cleaning
-----------------------

``plot`` draws the light curves in a grid of panels sharing their y limits, so transit depths can be
compared by eye. It is the entry point to the cleaning workflow.

.. code-block:: python

    lcs.plot(ncols=5, show_median=True, nsigma=3.0)
    lcs.plot(passbands='TESS', show_linear_model=True)

The panels can be filtered by passband, instrument, sector, or planet id, the running median can be
overlaid with its n-sigma clipping limits, and the variability explained by the covariates can be
drawn on top.

Bad light curves are removed in two steps: mark, then remove.

.. code-block:: python

    lcs.plot()                        # eyeball the light curves
    lcs.mark_for_removal([2, 7, 9])   # mark the bad ones by their panel index
    lcs.plot()                        # the marked ones now have a gray background
    lcs.remove_marked()

Marking only sets a flag. The marked light curves stay in the group and keep appearing in every bulk
property until ``remove_marked`` is called, so nothing a model sees changes until you commit. The
index shown in the upper left corner of each panel is exactly the index ``mark_for_removal`` takes.

.. warning::

    Marks are shared, removals are not. The flag lives in the `LCData` objects, which groups share
    by reference, so marking through a ``select`` result marks the light curve in the original group
    too. ``remove_marked`` drops the light curves from *that group only*.

    Remove before handing the group to a model: the removal renumbers every order-derived index,
    including `pbids` and `lcslices`.

Radial velocities
-----------------

`RVData` and `RVDataGroup` mirror the photometric containers, with two differences that follow from
the data:

- **Times are in days, velocities in m/s.**
- **The uncertainties are required.** Unlike photometry, a radial velocity measurement essentially
  always comes with an uncertainty, so `error` is a positional argument rather than an optional one.

.. code-block:: python

    from pytransit.utils.io import RVData

    rv = RVData(time, rv, error, instrument='HARPS')

The instrument name matters here: it identifies the dataset's systemic velocity and jitter
parameters, so it must be unique within a group.

API
---

Light curves
************

.. autoclass:: pytransit.utils.io.lcdata.LCData
    :members:
    :inherited-members:

.. autoclass:: pytransit.utils.io.lcdata.LCDataGroup
    :members:
    :inherited-members:

Radial velocities
*****************

.. autoclass:: pytransit.utils.io.rvdata.RVData
    :members:
    :inherited-members:

.. autoclass:: pytransit.utils.io.rvdata.RVDataGroup
    :members:
    :inherited-members:
