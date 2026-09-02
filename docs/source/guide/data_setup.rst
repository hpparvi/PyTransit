Data setup
==========

`set_data` tells the model *when* the observations were made and how they are organised. It is
called once per dataset, and everything it computes is reused by every later `evaluate` call.

.. code-block:: python

    tm.set_data(time, lcids=None, pbids=None, nsamples=None, exptimes=None, epids=None)

Only `time` is required. The remaining arguments describe the structure of a heterogeneous
dataset, and each of them can be ignored until you need it.

The simplest case
-----------------

If all the data were observed in one passband with an exposure time short enough that
supersampling is unnecessary, the mid-exposure times are all the model needs.

.. code-block:: python

    tm.set_data(time)

The model then treats the whole time array as a single light curve in a single passband.

The diagrams on this page all show one running example, larger than the snippets in the text: 25
exposures that will be split into three light curves. With only the times given, that is all the
model knows about them.

.. plot::
    :include-source: False

    from _figures.dataset_arrays import draw_dataset
    draw_dataset()

Each box is one exposure.


Light curves
------------

A PyTransit *light curve* is a group of exposures that share a passband, an exposure time, and a
supersampling rate. A time array may hold many of them: several nights of ground-based photometry,
several TESS sectors, or a mix of instruments.

To split the time array into light curves, pass an integer array `lcids` that maps each exposure
to a light curve.

.. code-block:: python

    tm.set_data(time=[0, 1, 2, 3, 4], lcids=[0, 0, 0, 1, 1])

`lcids` must have exactly as many elements as `time`. The number of light curves is inferred from
the number of unique values, so it never needs to be given explicitly.

.. plot::
    :include-source: False

    from _figures.dataset_arrays import draw_dataset
    draw_dataset(lcids=True)

In the running example the 25 exposures split into three light curves: the first ten, the next
five, and the last ten. `lcids` has one entry per exposure, so it lines up with `time` box for
box. The colours carry through the remaining diagrams, one per light curve.


Setting `lcids` alone changes nothing about the computed model. It is the scaffolding the other
arguments hang on: passbands, exposure times, and supersampling rates are all specified *per light
curve*.

Passbands
---------

Stellar limb darkening depends on wavelength, so transits observed in different passbands have
different shapes. Pass an integer array `pbids` that maps each *light curve* to a passband.

.. code-block:: python

    tm.set_data(time=[0, 1, 2, 3, 4], lcids=[0, 0, 0, 1, 1], pbids=[0, 1])

Note the change of granularity: `lcids` has one element per *exposure*, `pbids` one element per
*light curve*. Here, the two light curves belong to two different passbands.

Once more than one passband is defined, the model expects a set of limb darkening coefficients for
each of them, and optionally a radius ratio for each of them. See :doc:`evaluation`.

The passband indices must be a contiguous range starting at zero. Given three passbands, the valid
values are 0, 1, and 2; a set such as ``[0, 2]`` raises a `ValueError`.

.. plot::
    :include-source: False

    from _figures.dataset_arrays import draw_dataset
    draw_dataset(lcids=True, pbids=True)

`pbids` is short: one entry per *light curve*, not per exposure. Its three boxes are coloured to
match the light curves they describe. Light curves 0 and 2 are in passband 0, light curve 1 is in
passband 1 -- so this dataset has two passbands, and the model will expect two sets of limb
darkening coefficients.


Supersampling
-------------

A long exposure smears the transit signal: the measured flux is the average of the true flux over
the exposure, not its value at the mid-exposure time. Ignoring this rounds the sharp ingress and
egress corners of a long-cadence light curve and biases the inferred parameters.

PyTransit corrects for it by *supersampling*, evaluating the model at several points spread evenly
across each exposure and averaging the result. Give the exposure time `exptimes` in days and the
number of samples per exposure `nsamples`.

.. code-block:: python

    tm.set_data(time, exptimes=0.02, nsamples=10)

A single value applies to the whole time series. Arrays with one value per light curve let each
light curve have its own rate, which is what makes it possible to mix long- and short-cadence data
in one model.

.. code-block:: python

    tm.set_data(time=[0, 1, 2, 3, 4], lcids=[0, 0, 0, 1, 1],
                exptimes=[0.0007, 0.02], nsamples=[1, 10])

.. note::

    `exptimes` is in **days**, matching `time`. Kepler and TESS long cadence is roughly 0.02 d
    (~29 min), TESS 2-minute cadence is about 0.0014 d, and 20-second cadence about 0.00023 d.
    Supersampling rates of 10-20 are usually more than enough for long cadence; short-cadence data
    generally needs none.

.. plot::
    :include-source: False

    from _figures.dataset_arrays import draw_dataset
    draw_dataset(lcids=True, pbids=True, sampling=True)

`nsamples` and `exptimes` are per light curve as well. Light curve 0 is short cadence and needs no
supersampling; light curves 1 and 2 are long cadence and are integrated over ten samples each. This
is what lets a single model mix cadences.


Epochs
------

`epids` maps each light curve to an *epoch index*, which lets a light curve be linked to its own
zero epoch and period. This is what makes transit timing variation (TTV) modelling possible: each
transit gets an independent mid-transit time while everything else stays shared.

.. code-block:: python

    tm.set_data(time, lcids=lcids, pbids=pbids, epids=[0, 1, 2])

Like `pbids`, `epids` has one element per light curve. If it is not given, all the light curves
share a single epoch.

.. plot::
    :include-source: False

    from _figures.dataset_arrays import draw_dataset
    draw_dataset(lcids=True, pbids=True, epids=True, sampling=True)

The complete picture. Three arrays are indexed by exposure (`time`, and `lcids` which maps them to
light curves), and four are indexed by light curve (`pbids`, `epids`, `nsamples`, `exptimes`).
Getting that distinction right is most of what there is to learn about `set_data`.


A worked example
----------------

Consider three light curves: two observed in one passband and a third in another, one of them at
long cadence.

.. code-block:: text

    times_1  (lc = 0, pb = 0, short cadence) = [1, 2, 3, 4]
    times_2  (lc = 1, pb = 0, long cadence)  = [3, 4]
    times_3  (lc = 2, pb = 1, short cadence) = [1, 5, 6]

They are concatenated into a single time array and described by the index arrays

.. code-block:: python

    tm.set_data(time     = [1, 2, 3, 4, 3, 4, 1, 5, 6],
                lcids    = [0, 0, 0, 0, 1, 1, 2, 2, 2],
                pbids    = [0, 0, 1],
                nsamples = [1, 10, 1],
                exptimes = [0.1, 1.0, 0.1])

Reading the arrays: `time` and `lcids` have nine elements each, one per exposure. `pbids`,
`nsamples`, and `exptimes` have three elements each, one per light curve. Light curves 0 and 1 are
in passband 0, light curve 2 is in passband 1, and light curve 1 is supersampled ten times.

The times need not be sorted, and light curves may overlap in time -- as light curves 0 and 1 do
here, which is exactly the situation when the same transit is observed simultaneously by two
instruments.

Validation
----------

`set_data` checks its arguments and raises a `ValueError` with an explanatory message when they
are inconsistent. The rules are:

- `lcids` and `pbids` must have an integer dtype. Passing floats, which is easy to do accidentally
  via ``numpy.zeros(n)``, is rejected rather than silently truncated.
- ``lcids.size`` must equal ``time.size``.
- ``pbids.size`` must equal the number of unique light curve indices.
- `pbids` must span a contiguous range from 0 to ``npb - 1``.

.. warning::

    `set_data` skips its work if it is called again with the *same time array object* and no other
    arguments. The check is on the object's identity, so

    .. code-block:: python

        tm.set_data(time, lcids=lcids, pbids=pbids)
        tm.set_data(time)                            # does not reset lcids and pbids

    leaves the earlier `lcids` and `pbids` in place instead of reverting to the single-light-curve
    default. To genuinely reset the setup, pass a distinct array object, for example
    ``tm.set_data(time.copy())``.

What the model stores
---------------------

After `set_data` returns, the model exposes the parsed setup as attributes, which are useful when
debugging a dataset.

.. list-table::
    :header-rows: 1
    :widths: 18 82

    * - Attribute
      - Contents
    * - ``time``
      - Mid-exposure times as a ``float64`` array.
    * - ``lcids``
      - Light curve index per exposure.
    * - ``pbids``
      - Passband index per light curve.
    * - ``epids``
      - Epoch index per light curve.
    * - ``nsamples``
      - Supersampling rate per light curve.
    * - ``exptimes``
      - Exposure time per light curve.
    * - ``npt``
      - Number of exposures.
    * - ``nlc``
      - Number of light curves.
    * - ``npb``
      - Number of passbands.

.. seealso::

    :doc:`../io` describes `LCData` and `LCDataGroup`, containers that hold light curves together
    with their metadata. A group exposes `pbids`, `exptimes`, `nsamples`, and `lcslices` as bulk
    properties, so it can supply most of these arrays directly.
