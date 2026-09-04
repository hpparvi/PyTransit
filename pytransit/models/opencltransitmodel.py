#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2026  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List, Optional, Union

import pyopencl as cl
from numpy import asarray, atleast_1d, float32, ndarray, uint32

from .transitmodel import TransitModel

__all__ = ['OpenCLTransitModel']


class OpenCLTransitModel(TransitModel):
    """Base class for the OpenCL transit models.

    Holds the data setup shared by every OpenCL model. `TransitModel.set_data` does the
    validation and the bookkeeping, and this class casts the arrays into the types the kernels
    index and uploads them.

    Each OpenCL model used to carry its own copy of `set_data`, which is how the four copies
    came to disagree with the base class about the default exposure time, and how none of them
    ended up validating the light curve and passband indices at all.
    """

    # The device floating point type of the time and the exposure times. A model that offers a
    # choice of precision overrides it per instance.
    dtype = float32

    # The buffers holding the data. Declared here so that `set_data` can release them on a
    # subsequent call whether or not the subclass has initialised them.
    _b_time = None
    _b_lcids = None
    _b_pbids = None
    _b_nsamples = None
    _b_etimes = None

    _data_buffers = ('_b_time', '_b_lcids', '_b_pbids', '_b_nsamples', '_b_etimes')

    def set_data(self, time: Union[ndarray, List],
                 lcids: Optional[Union[ndarray, List]] = None,
                 pbids: Optional[Union[ndarray, List]] = None,
                 nsamples: Optional[Union[ndarray, List]] = None,
                 exptimes: Optional[Union[ndarray, List]] = None,
                 epids: Optional[Union[ndarray, List]] = None) -> bool:
        """Set the data and upload it to the device.

        Takes the same arguments as `TransitModel.set_data`, which does the validation and stores
        the arrays. They are then cast to the types the kernels index -- the times and the exposure
        times to the device floating point type, the indices and the sample counts to `uint32` --
        and uploaded. Nothing happens if the base class reports the call a no-op.
        """
        if not super().set_data(time, lcids, pbids, nsamples, exptimes, epids):
            return False

        # The kernels index these, and take the counts as scalar arguments, so they have to carry
        # the device types rather than the host ones.
        self.time = asarray(self.time, self.dtype)
        self.lcids = asarray(self.lcids, uint32)
        self.pbids = asarray(self.pbids, uint32)
        self.nsamples = atleast_1d(asarray(self.nsamples, uint32))
        self.exptimes = atleast_1d(asarray(self.exptimes, self.dtype))
        self.nlc = uint32(self.nlc)
        self.npb = uint32(self.npb)
        self.nptb = self.npt

        for name in self._data_buffers:
            buffer = getattr(self, name)
            if buffer is not None:
                buffer.release()

        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        self._b_time = cl.Buffer(self.ctx, flags, hostbuf=self.time)
        self._b_lcids = cl.Buffer(self.ctx, flags, hostbuf=self.lcids)
        self._b_pbids = cl.Buffer(self.ctx, flags, hostbuf=self.pbids)
        self._b_nsamples = cl.Buffer(self.ctx, flags, hostbuf=self.nsamples)
        self._b_etimes = cl.Buffer(self.ctx, flags, hostbuf=self.exptimes)

        self._on_data_set()
        return True

    def _on_data_set(self) -> None:
        """Invalidate whatever the model derives from the data. Does nothing by default."""
