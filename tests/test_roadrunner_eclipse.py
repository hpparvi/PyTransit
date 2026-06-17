#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

from math import pi

import numpy as np
import pytest
from numpy import ndarray


def make_inputs(k=0.1, a=8.0, inc=np.pi / 2, p=2.0, t0=0.0,
                e=0.0, w=0.0, rstar=1.0,
                npt=5000, t_start=-0.5, t_end=3.5, nep=1):
    """Build a canonical single-pv, single-light-curve input set.

    The defaults give a non-grazing hot-Jupiter-like geometry on a
    circular orbit, with the primary transit at t=0 and the secondary
    eclipse near t=p/2=1 d.
    """
    times = np.linspace(t_start, t_end, npt)
    k_arr = np.array([k])
    t0_arr = np.full((1, nep), t0)
    p_arr = np.array([p])
    a_arr = np.array([a])
    i_arr = np.array([inc])
    e_arr = np.array([e])
    w_arr = np.array([w])
    rstar = float(rstar)
    nlc = 1
    lcids = np.zeros(npt, dtype=np.int64)
    epids = np.zeros(nlc, dtype=np.int64)
    nsamples = np.ones(nlc, dtype=np.int64)
    exptimes = np.zeros(nlc)
    return (times, k_arr, t0_arr, p_arr, a_arr, i_arr, e_arr, w_arr,
            rstar, nlc, lcids, epids, nsamples, exptimes)


class TestEclipseKernelImport:
    """Sanity checks that the kernel imports and runs end-to-end."""

    def test_kernel_imports(self):
        from pytransit.models.roadrunner.model_eclipse import eclipse_model
        assert callable(eclipse_model)

    def test_kernel_returns_expected_shape(self):
        from pytransit.models.roadrunner.model_eclipse import eclipse_model
        args = make_inputs()
        flux = eclipse_model(*args)
        assert isinstance(flux, ndarray)
        assert flux.shape == (1, args[0].size)

    def test_kernel_produces_eclipse_dip(self):
        from pytransit.models.roadrunner.model_eclipse import eclipse_model
        args = make_inputs()
        times, k = args[0], args[1][0]
        flux = eclipse_model(*args)[0]
        baseline = pi * k ** 2
        # Out of eclipse the kernel returns the full (unocculted) planet area.
        assert flux.max() == pytest.approx(baseline, abs=1e-6)
        # The secondary eclipse near t = p/2 = 1 d occults the planet, dropping the area.
        i_ecl = int(np.argmin(np.abs(times - 1.0)))
        assert flux[i_ecl] < baseline - 1e-4
        # A non-grazing eclipse is total, so the minimum area reaches ~0.
        assert flux.min() < 1e-3
