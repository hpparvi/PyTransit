Deprecated evaluation API
=========================

.. warning::

    Everything on this page is **deprecated since PyTransit 2.9 and will be removed in PyTransit
    3.0**. It is documented here only so that existing code can be migrated. New code should use
    `evaluate`, described in :doc:`../guide/evaluation`.

Before 2.9, evaluating a transit model meant choosing between three methods:

``evaluate_ps(...)``
    Evaluate for a single *parameter set* of scalars.

``evaluate_pv(pvp, ldc)``
    Evaluate for a *parameter vector population*: a 2D array with one row per parameter vector,
    the parameters packed in the order ``[k, t0, p, a, i, e, w]``.

``evaluate(...)``
    Dispatch to one of the above based on the types of the arguments.

`evaluate` now does all of it directly. It accepts scalars, 1D arrays, and 2D arrays and
broadcasts on their shapes, so the split serves no purpose.

Calling any method on this page raises an
:class:`~astropy.utils.exceptions.AstropyDeprecationWarning`. The methods still work and still
return the same results; only the warning is new.

Migrating
---------

**From ``evaluate_ps``** -- the arguments are unchanged, so the call is the same with a different
method name:

.. code-block:: python

    flux = tm.evaluate_ps(k, ldc, t0, p, a, i, e, w)      # before
    flux = tm.evaluate(k, ldc, t0, p, a, i, e, w)         # after

**From ``evaluate_pv``** -- unpack the columns of the packed parameter array into named arguments.
Note that ``k`` must be 2D with shape ``(npv, npb)``, which is what column zero already is when
sliced with a list:

.. code-block:: python

    flux = tm.evaluate_pv(pvp, ldc)                                    # before
    flux = tm.evaluate(k=pvp[:, [0]], ldc=ldc, t0=pvp[:, 1],           # after
                       p=pvp[:, 2], a=pvp[:, 3], i=pvp[:, 4],
                       e=pvp[:, 5], w=pvp[:, 6])

**From ``evaluate_pv_ttv``** -- there is no replacement yet. The OpenCL models cannot express
per-epoch transit centres through `evaluate`, because the OpenCL `set_data` does not accept
`epids`. Use the Numba :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` with `epids`
for TTV work, or keep using this method until a replacement lands.

Silencing the warnings
----------------------

While migrating, the warnings can be suppressed:

.. code-block:: python

    import warnings
    from astropy.utils.exceptions import AstropyDeprecationWarning

    warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)

.. note::

    Some of PyTransit's own log posterior function classes still call ``evaluate_pv`` internally,
    so running one emits these warnings until they are migrated too.

Numba models
------------

.. automethod:: pytransit.models.ma_quadratic.QuadraticModel.evaluate_ps

.. automethod:: pytransit.models.ma_uniform.UniformModel.evaluate_ps

.. automethod:: pytransit.models.eclipse_model.EclipseModel.evaluate_ps

.. automethod:: pytransit.models.qpower2.QPower2Model.evaluate_ps

.. automethod:: pytransit.models.ma_chromosphere.ChromosphereModel.evaluate_ps

.. automethod:: pytransit.models.general.GeneralModel.evaluate_ps

.. automethod:: pytransit.models.general.GeneralModel.evaluate_pv

.. automethod:: pytransit.models.gdmodel.GravityDarkenedModel.evaluate_ps

OpenCL models
-------------

.. automethod:: pytransit.models.ma_uniform_cl.UniformModelCL.evaluate_ps

.. automethod:: pytransit.models.ma_uniform_cl.UniformModelCL.evaluate_pv

.. automethod:: pytransit.models.ma_quadratic_cl.QuadraticModelCL.evaluate_ps

.. automethod:: pytransit.models.ma_quadratic_cl.QuadraticModelCL.evaluate_pv

.. automethod:: pytransit.models.ma_quadratic_cl.QuadraticModelCL.evaluate_pv_ttv

.. automethod:: pytransit.models.qpower2_cl.QPower2ModelCL.evaluate_ps

.. automethod:: pytransit.models.qpower2_cl.QPower2ModelCL.evaluate_pv

.. automethod:: pytransit.models.roadrunner.rrmodel_cl.RoadRunnerModelCL.evaluate_ps

.. automethod:: pytransit.models.roadrunner.rrmodel_cl.RoadRunnerModelCL.evaluate_pv
