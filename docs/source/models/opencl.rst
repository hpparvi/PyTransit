OpenCL models
=============

The GPU implementations. See :doc:`../guide/opencl` for how they are used, how they differ from the
CPU models, and when the GPU is worth it.

.. warning::

    The OpenCL models compute in **single precision**. Always subtract a constant epoch from the
    times before passing them to one.

Uniform model
-------------

.. autoclass:: pytransit.models.ma_uniform_cl.UniformModelCL
    :members: set_data, evaluate
    :special-members: __init__

Quadratic model
---------------

.. autoclass:: pytransit.models.ma_quadratic_cl.QuadraticModelCL
    :members: set_data, evaluate
    :special-members: __init__

Power-2 model
-------------

.. autoclass:: pytransit.models.qpower2_cl.QPower2ModelCL
    :members: set_data, evaluate
    :special-members: __init__

RoadRunner model
----------------

.. autoclass:: pytransit.models.roadrunner.rrmodel_cl.RoadRunnerModelCL
    :members: set_data, evaluate, init_integration
    :special-members: __init__
