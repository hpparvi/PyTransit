"""Gimenez transit model

   Includes a class offering an easy access to the Fortran implementation of the
   transit model by A. Gimenez (A&A 450, 1231--1237, 2006).

   Author
     Hannu Parviainen <hpparvi@gmail.com>
"""
import pyopencl as cl
import numpy as np

READ_ONLY, WRITE_ONLY = cl.mem_flags.READ_ONLY, cl.mem_flags.WRITE_ONLY
READ_WRITE, COPY_HOST_PTR = cl.mem_flags.READ_WRITE, cl.mem_flags.COPY_HOST_PTR

from numpy import float32 as f32

class Orbit(object):
    def __init__(self, ctx=None, queue=None):
        self.ctx = ctx or cl.create_some_context()
        self.queue = queue or cl.CommandQueue(self.ctx)

        prg = cl.Program(self.ctx, """
            __kernel void z_circular(__global const float *t, 
                 const float t0, const float p, 
                 const float a, const float i,
                 __global float *z)
            {
              int gid = get_global_id(0);

              float n = M_2_PI_F/p;
              float sini  = sin(i);
              float cosph = cos((t[gid]-t0)*n);

              z[gid] = a * sqrt(1. -  cosph*cosph*sini*sini);
            }
            """).build()

        self.k_z_circular = prg.z_circular


    def z_circular(self, t, t0, p, a, i, update_t=True):
        if True: #update_t:
            t = t.astype(f32)
            self.t_buf = cl.Buffer(self.ctx, READ_ONLY | COPY_HOST_PTR, hostbuf=t)
            self.z = np.zeros_like(t)
            self.z_buf = cl.Buffer(self.ctx, WRITE_ONLY, t.nbytes)

        self.k_z_circular(self.queue, t.shape, None, self.t_buf, f32(t0), f32(p), f32(a), f32(i), self.z_buf)
        cl.enqueue_copy(self.queue, self.z, self.z_buf)

        return self.z
