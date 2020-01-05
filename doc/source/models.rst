Transit model interface
=======================

OpenCL
------

The OpenCL versions of the models work identically to the Python version, except
that the OpenCL context and queue can be given as arguments in the initialiser, and the model evaluation method can be
told to not to copy the model from the GPU memory. If the context and queue are not given, the model creates a default
context using `cl.create_some_context()`.

Python
import pyopencl as cl
from src import QuadraticModelCL

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

tm = QuadraticModelCL(cl_ctx=ctx, cl_queue=queue)


Transit models
==============

PyTransit implements a set of transit models that all share a common API.

Uniform model
-------------

Quadratic model
---------------

Power-2 model
-------------

Power-2 transit model by Maxted & Gill (A&A 622, A33 2019).

Gimenez model
-------------

Chromosphere model
------------------

Optically thin shell model by Schlawin et al. (ApJL 722, 75--79, 2010) to model a transit over a chromosphere.