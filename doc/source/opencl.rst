OpenCL
------

The OpenCL versions of the models work identically to the Python version, except
that the OpenCL context and queue can be given as arguments in the initialiser, and the model evaluation method can be
told to not to copy the model from the GPU memory. If the context and queue are not given, the model creates a default
context using `cl.create_some_context()`.

```Python
import pyopencl as cl
from src import QuadraticModelCL

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

tm = QuadraticModelCL(cl_ctx=ctx, cl_queue=queue)
```