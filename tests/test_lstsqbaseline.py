#  PyTransit: fast and easy exoplanet transit modelling in Python.
#  Copyright (C) 2010-2019  Hannu Parviainen
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

import pytest
from numpy import (linspace, ones, zeros, array, isfinite, isnan, allclose, concatenate,
                   atleast_2d, tile, newaxis)
from numpy.random import default_rng
from numpy.testing import assert_allclose

from pytransit import BaseLPF, LSTSQBaseline, RoadRunnerModel

NLC = 3
NPT = 120
NCOV = 2


def make_data(seed: int = 0):
    """Three light curves with a known linear-in-covariates baseline and a transit."""
    rng = default_rng(seed)
    times, covs, baselines = [], [], []
    for i in range(NLC):
        t = linspace(0.9 + i, 1.1 + i, NPT)
        cv = rng.normal(0.0, 1.0, (NPT, NCOV))
        coefs = array([1.0 + 0.01 * i, 0.004, -0.002])
        bl = coefs[0] + cv @ coefs[1:]
        times.append(t)
        covs.append(cv)
        baselines.append(bl)
    return times, covs, baselines


class LSQLPF(BaseLPF):
    """A minimal LPF using LSTSQBaseline via the documented flux_model override."""

    def _init_baseline(self):
        self.lstsq_baseline = LSTSQBaseline(self)

    def flux_model(self, pv):
        fmod = self.transit_model(pv)
        return fmod * self.lstsq_baseline(fmod) + self.trends(pv)


def make_lpf(seed: int = 0):
    times, covs, baselines = make_data(seed)
    # Placeholder fluxes; overwritten below once the true transit model is known.
    fluxes = [ones(NPT) for _ in range(NLC)]
    lpf = LSQLPF('test', ['g'], times=times, fluxes=fluxes, covariates=covs,
                 tm=RoadRunnerModel('quadratic'))
    pv = lpf.ps.mean_pv.copy()
    pv[0], pv[1], pv[2], pv[3] = 1.0, 2.5, 2.0, 0.1
    pv[4] = 0.01
    ftr = lpf.transit_model(pv)
    blc = concatenate(baselines)
    lpf.ofluxa[:] = ftr * blc
    for i, sl in enumerate(lpf.lcslices):
        lpf.fluxes[i][:] = lpf.ofluxa[sl]
    return lpf, pv, ftr, blc


class TestRecovery:
    def test_recovers_injected_baseline(self):
        """Given the true transit model, the fit must return the injected baseline."""
        lpf, pv, ftr, blc = make_lpf()
        assert_allclose(lpf.lstsq_baseline(ftr), blc, rtol=1e-10)

    def test_flux_model_reproduces_data(self):
        lpf, pv, ftr, blc = make_lpf()
        assert_allclose(lpf.flux_model(pv), lpf.ofluxa, rtol=1e-10)

    def test_coefficients_match_injected(self):
        """Coefficients live in standardised covariate space; mapping them back to the raw
        space must recover the injected values exactly. This verifies that standardisation
        is an exact reparameterisation rather than a change of model."""
        lpf, pv, ftr, blc = make_lpf()
        _, raw_covs, _ = make_data()
        cs = lpf.lstsq_baseline.coefficients(ftr)
        assert len(cs) == NLC
        for i, c in enumerate(cs):
            m, sd = raw_covs[i].mean(0), raw_covs[i].std(0)
            slopes = c[1:] / sd
            intercept = c[0] - m @ slopes
            assert_allclose(concatenate([[intercept], slopes]),
                            array([1.0 + 0.01 * i, 0.004, -0.002]), atol=1e-10)


class TestNormalization:
    def test_lpf_standardizes_covariates(self):
        """BaseLPF._init_data must actually write the standardised covariates back."""
        times, covs, _ = make_data()
        covs = [c * 1e4 + 2.46e6 for c in covs]     # wildly unnormalised
        lpf = LSQLPF('norm', ['g'], times=times, fluxes=[ones(NPT)] * NLC,
                     covariates=covs, tm=RoadRunnerModel('quadratic'))
        for cv in lpf.covariates:
            assert_allclose(cv.mean(0), 0.0, atol=1e-10)
            assert_allclose(cv.std(0), 1.0, atol=1e-10)

    def test_standardization_is_idempotent(self):
        times, covs, _ = make_data()
        once = LSTSQBaseline.normalize_covariates(covs[0])
        twice = LSTSQBaseline.normalize_covariates(once)
        assert_allclose(once, twice, atol=1e-12)

    def test_unnormalised_covariates_still_recover_baseline(self):
        """A badly scaled covariate must not degrade the fit, since it is standardised."""
        rng = default_rng(3)
        t = linspace(0.9, 1.1, NPT)
        cv = zeros((NPT, 2))
        cv[:, 0] = 2.46e6 + rng.normal(0, 1e-2, NPT)    # raw-BJD-like column
        cv[:, 1] = rng.normal(0, 1, NPT)
        bl = 1.0 + 3e-9 * (cv[:, 0] - cv[:, 0].mean()) - 0.002 * cv[:, 1]
        lpf = LSQLPF('scaled', ['g'], times=[t], fluxes=[ones(NPT)], covariates=[cv],
                     tm=RoadRunnerModel('quadratic'))
        pv = lpf.ps.mean_pv.copy()
        pv[0], pv[1], pv[2], pv[3], pv[4] = 1.0, 2.5, 2.0, 0.1, 0.01
        ftr = lpf.transit_model(pv)
        lpf.ofluxa[:] = ftr * bl
        assert_allclose(lpf.lstsq_baseline(ftr), bl, rtol=1e-10)

    def test_constant_covariate_column_raises(self):
        times, covs, _ = make_data()
        covs = [c.copy() for c in covs]
        covs[0][:, 0] = 7.0
        with pytest.raises(ValueError, match='singular or badly conditioned'):
            LSQLPF('const', ['g'], times=times, fluxes=[ones(NPT)] * NLC,
                   covariates=covs, tm=RoadRunnerModel('quadratic'))


class TestParameterless:
    def test_adds_no_parameters(self):
        """The whole point: the baseline must not grow the parameter set."""
        lpf, _, _, _ = make_lpf()
        times, covs, _ = make_data()
        ref = BaseLPF('ref', ['g'], times=times, fluxes=[ones(NPT)] * NLC,
                      covariates=covs, tm=RoadRunnerModel('quadratic'))
        assert len(lpf.ps) == len(ref.ps)

    def test_no_baseline_block_in_ps(self):
        lpf, _, _, _ = make_lpf()
        assert not any('lstsq' in p.name for p in lpf.ps)


class _DEStub:
    """Stands in for the DE optimiser so remove_outliers can be tested without running it."""

    def __init__(self, pv):
        self.minimum_location = pv


class TestRemoveOutliers:
    def test_covariates_are_masked_with_the_data(self):
        """remove_outliers must pass the *masked* covariates to _init_data, not the originals."""
        lpf, pv, ftr, blc = make_lpf()
        lpf.ofluxa[7] += 0.05
        lpf.fluxes[0][7] += 0.05
        lpf.de = _DEStub(pv)
        npt_before = lpf.timea.size

        lpf.remove_outliers(sigma=4)

        assert lpf.timea.size < npt_before
        for i in range(lpf.nlc):
            assert lpf.covariates[i].shape[0] == lpf.times[i].size

    def test_covariates_stay_standardised_after_clipping(self):
        lpf, pv, ftr, blc = make_lpf()
        lpf.ofluxa[7] += 0.05
        lpf.fluxes[0][7] += 0.05
        lpf.de = _DEStub(pv)
        lpf.remove_outliers(sigma=4)
        for cv in lpf.covariates:
            assert_allclose(cv.mean(0), 0.0, atol=1e-10)
            assert_allclose(cv.std(0), 1.0, atol=1e-10)


class TestShapes:
    def test_1d_in_1d_out(self):
        lpf, pv, ftr, blc = make_lpf()
        assert lpf.lstsq_baseline(ftr).shape == (NLC * NPT,)

    def test_2d_in_2d_out(self):
        lpf, pv, ftr, blc = make_lpf()
        fm = tile(ftr, (4, 1))
        assert lpf.lstsq_baseline(fm).shape == (4, NLC * NPT)

    def test_batch_matches_single(self):
        lpf, pv, ftr, blc = make_lpf()
        fm = tile(ftr, (3, 1)) * array([1.0, 1.001, 0.999])[:, newaxis]
        batched = lpf.lstsq_baseline(fm).copy()
        for i in range(3):
            assert_allclose(batched[i], lpf.lstsq_baseline(fm[i]), rtol=1e-12)


class TestObservedFlux:
    def test_explicit_oflux_matches_default(self):
        lpf, pv, ftr, blc = make_lpf()
        assert_allclose(lpf.lstsq_baseline(ftr, lpf.ofluxa), lpf.lstsq_baseline(ftr), rtol=1e-12)

    def test_scaled_oflux_scales_the_baseline(self):
        lpf, pv, ftr, blc = make_lpf()
        bl = lpf.lstsq_baseline(ftr, 2.0 * lpf.ofluxa)
        assert_allclose(bl, 2.0 * blc, rtol=1e-10)

    def test_2d_oflux_fits_each_row_separately(self):
        """A 2D oflux lets every parameter vector see its own dataset."""
        lpf, pv, ftr, blc = make_lpf()
        factors = array([1.0, 2.0, 0.5])
        of = lpf.ofluxa * factors[:, newaxis]
        bl = lpf.lstsq_baseline(tile(ftr, (3, 1)), of)
        assert_allclose(bl, blc * factors[:, newaxis], rtol=1e-10)

    def test_1d_oflux_broadcasts_over_population(self):
        lpf, pv, ftr, blc = make_lpf()
        bl = lpf.lstsq_baseline(tile(ftr, (3, 1)), lpf.ofluxa)
        assert_allclose(bl, tile(blc, (3, 1)), rtol=1e-10)

    def test_coefficients_accept_oflux(self):
        lpf, pv, ftr, blc = make_lpf()
        cs = lpf.lstsq_baseline.coefficients(ftr, 2.0 * lpf.ofluxa)
        ref = lpf.lstsq_baseline.coefficients(ftr)
        for c, r in zip(cs, ref):
            assert_allclose(c, 2.0 * r, rtol=1e-10)

    def test_wrong_size_oflux_raises(self):
        lpf, pv, ftr, blc = make_lpf()
        with pytest.raises(ValueError, match='observed flux has'):
            lpf.lstsq_baseline(ftr, lpf.ofluxa[:-1])


class TestFailureMode:
    def test_zero_flux_gives_nan_for_affected_light_curve(self):
        """NaN is localised to the light curve that failed, as in ExoIris."""
        lpf, pv, ftr, blc = make_lpf()
        fm = tile(ftr, (3, 1))
        fm[1, 5] = 0.0                      # point 5 belongs to light curve 0
        bl = lpf.lstsq_baseline(fm)
        assert isnan(bl[1, lpf.lcslices[0]]).all()
        assert isfinite(bl[1, lpf.lcslices[1]]).all()
        assert isfinite(bl[1, lpf.lcslices[2]]).all()
        assert isfinite(bl[0]).all()
        assert isfinite(bl[2]).all()

    def test_nan_baseline_gives_nonfinite_lnposterior(self):
        lpf, pv, ftr, blc = make_lpf()
        lpf.ofluxa[3] = float('nan')
        assert not isfinite(lpf.lnposterior(pv))

    def test_singular_design_matrix_raises(self):
        times, covs, _ = make_data()
        covs = [c.copy() for c in covs]
        covs[1][:, 1] = covs[1][:, 0]  # duplicate column -> rank deficient
        with pytest.raises(ValueError, match='singular or badly conditioned'):
            LSQLPF('sing', ['g'], times=times, fluxes=[ones(NPT)] * NLC,
                   covariates=covs, tm=RoadRunnerModel('quadratic'))

    def test_missing_covariates_raises(self):
        times, _, _ = make_data()
        with pytest.raises(ValueError, match='requires the LPF to be initialised with covariates'):
            LSQLPF('nocov', ['g'], times=times, fluxes=[ones(NPT)] * NLC,
                   tm=RoadRunnerModel('quadratic'))


class TestBuffer:
    def test_no_stale_values_across_npv_change(self):
        lpf, pv, ftr, blc = make_lpf()
        lpf.lstsq_baseline(tile(ftr, (5, 1)))
        bl = lpf.lstsq_baseline(tile(ftr, (2, 1)))
        assert bl.shape == (2, NLC * NPT)
        assert_allclose(bl, tile(blc, (2, 1)), rtol=1e-10)

    def test_uncovered_points_stay_unity(self):
        lpf, pv, ftr, blc = make_lpf()
        blm = LSTSQBaseline(lpf, lcids=array([0]))
        bl = blm(ftr)
        assert_allclose(bl[lpf.lcslices[0]], blc[lpf.lcslices[0]], rtol=1e-10)
        assert_allclose(bl[lpf.lcslices[1]], 1.0)
        assert_allclose(bl[lpf.lcslices[2]], 1.0)


class TestEquivalence:
    def test_matches_linear_model_baseline(self):
        """LSTSQBaseline coefficients must be the LinearModelBaseline optimum."""
        from pytransit import LinearModelBaseline
        from scipy.optimize import minimize

        lpf, pv, ftr, blc = make_lpf()

        class LMLPF(BaseLPF):
            def _init_baseline(self):
                self._add_baseline_model(LinearModelBaseline(self))

        times, covs, _ = make_data()
        lm = LMLPF('lm', ['g'], times=times, fluxes=[lpf.ofluxa[sl].copy() for sl in lpf.lcslices],
                   covariates=covs, tm=RoadRunnerModel('quadratic'))

        pv_lm = lm.ps.mean_pv.copy()
        pv_lm[:5] = pv[:5]
        sl = lm._sl_lm

        def nll(c):
            p = pv_lm.copy()
            p[sl] = c
            return ((lm.ofluxa - lm.flux_model(p)) ** 2).sum()

        res = minimize(nll, pv_lm[sl], method='Nelder-Mead',
                       options=dict(maxiter=50000, xatol=1e-10, fatol=1e-14))
        expected = concatenate(lpf.lstsq_baseline.coefficients(ftr))
        assert_allclose(res.x, expected, atol=1e-4)
