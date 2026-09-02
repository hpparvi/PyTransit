TransitModel
============

The base class every PyTransit transit model derives from. It implements `set_data`, which is
therefore shared by all the models, and declares the `evaluate` interface the subclasses implement.

See :doc:`../guide/interface` and :doc:`../guide/data_setup` for the narrative description.

.. autoclass:: pytransit.models.transitmodel.TransitModel
    :members:
    :special-members: __init__

Limb darkening model interface
------------------------------

`LDModel` is the interface for limb darkening profiles that come from a stellar atmosphere model
rather than an analytic law. :class:`~pytransit.models.roadrunner.rrmodel.RoadRunnerModel` accepts
an instance of a subclass in place of a law name.

.. autoclass:: pytransit.models.ldmodel.LDModel
    :members:
    :special-members: __init__, __call__

.. autoclass:: pytransit.models.ldtkldm.LDTkLDModel
    :members:
    :special-members: __init__, __call__
