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

import pytest
from numpy import (linspace, ones, zeros, full, array, arange, diff, nanstd, sqrt, nan,
                   isnan, isfinite, float64, pi)
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal

from pytransit.lpf.rvlpf import RVLPF
from pytransit.utils.io import LightCurveData, RVData, RVDataGroup

NPT = 40


def make_rv(npt=NPT, t0=0.0, ncov=2, instrument='HARPS', seed=0):
    rng = default_rng(seed)
    t = linspace(t0, t0 + 100.0, npt)
    rv = 10.0 * rng.normal(0.0, 1.0, npt)
    e = full(npt, 2.0)
    cv = rng.normal(0.0, 1.0, (npt, ncov)) if ncov else None
    return RVData(t, rv, e, covariates=cv, instrument=instrument)


def make_group():
    return RVDataGroup([make_rv(instrument='HARPS', seed=0),
                        make_rv(instrument='HARPS-N', t0=200.0, seed=1),
                        make_rv(instrument='CARMENES', t0=400.0, seed=2)])


class TestCoercion:
    def test_lists_become_float64_arrays(self):
        d = RVData([1.0, 2.0, 3.0], [1, 2, 3], [1, 1, 1])
        assert d.time.dtype == float64 and d.rv.dtype == float64 and d.error.dtype == float64

    def test_int_arrays_become_float(self):
        d = RVData(arange(10), arange(10), ones(10, int))
        assert d.rv.dtype == float64

    def test_sizes(self):
        d = make_rv(ncov=3)
        assert d.size == NPT and d.npt == NPT and d.ncov == 3

    def test_covariates_default_to_empty_matrix(self):
        d = RVData(arange(10), ones(10), ones(10))
        assert d.covariates.shape == (10, 0) and d.ncov == 0

    def test_1d_covariates_become_a_single_column(self):
        d = RVData(arange(10), ones(10), ones(10), covariates=arange(10))
        assert d.covariates.shape == (10, 1)

    def test_arrays_are_not_copied(self):
        t, rv, e = linspace(0, 1, 10), ones(10), ones(10)
        d = RVData(t, rv, e)
        assert d.time is t and d.rv is rv and d.error is e


class TestValidation:
    def test_rv_length_mismatch(self):
        with pytest.raises(ValueError, match="'rv' has 9 points but 'time' has 10"):
            RVData(arange(10), ones(9), ones(10))

    def test_error_length_mismatch(self):
        with pytest.raises(ValueError, match="'error' has 9 points but 'time' has 10"):
            RVData(arange(10), ones(10), ones(9))

    def test_2d_rv_raises(self):
        with pytest.raises(ValueError, match="'rv' must be a 1D array"):
            RVData(arange(10), ones((10, 2)), ones(10))

    def test_2d_error_raises(self):
        with pytest.raises(ValueError, match="'error' must be a 1D array"):
            RVData(arange(10), ones(10), ones((10, 2)))

    def test_nonpositive_error_raises(self):
        e = ones(10)
        e[3] = 0.0
        with pytest.raises(ValueError, match="'error' contains non-positive values"):
            RVData(arange(10), ones(10), e)

    def test_nonfinite_time_raises(self):
        t = linspace(0, 1, 10)
        t[2] = nan
        with pytest.raises(ValueError, match="'time' contains non-finite values"):
            RVData(t, ones(10), ones(10))

    def test_3d_covariates_raises(self):
        with pytest.raises(ValueError, match="'covariates' must be a 1D or 2D array"):
            RVData(arange(10), ones(10), ones(10), covariates=zeros((10, 2, 2)))

    def test_nonfinite_rv_warns(self):
        rv = ones(10)
        rv[2] = nan
        with pytest.warns(UserWarning, match="'rv' contains non-finite values"):
            RVData(arange(10), rv, ones(10))

    def test_nonfinite_error_warns(self):
        e = ones(10)
        e[2] = nan
        with pytest.warns(UserWarning, match="'error' contains non-finite values"):
            RVData(arange(10), ones(10), e)

    def test_missing_error_raises(self):
        with pytest.raises(TypeError):
            RVData(arange(10), ones(10))

    def test_explicit_none_error_raises(self):
        with pytest.raises(ValueError, match="'error' is required for RV data"):
            RVData(arange(10), ones(10), None)

    def test_pids_is_not_accepted(self):
        """RV data has no planet subset: every planet contributes to the signal."""
        with pytest.raises(TypeError):
            RVData(arange(10), ones(10), ones(10), pids=[0])


class TestNoise:
    def test_noise_is_the_point_to_point_scatter(self):
        rng = default_rng(0)
        rv = rng.normal(0, 5.0, NPT)
        d = RVData(linspace(0, 100, NPT), rv, ones(NPT))
        assert_allclose(d.noise, nanstd(diff(rv)) / sqrt(2))

    def test_single_point_noise_is_nan(self):
        assert isnan(RVData([1.0], [1.0], [1.0]).noise)

    def test_has_error_is_always_true(self):
        assert make_rv().has_error


class TestAddition:
    def test_rv_plus_rv(self):
        a, b = make_rv(seed=0), make_rv(seed=1, instrument='HARPS-N')
        g = a + b
        assert isinstance(g, RVDataGroup) and g.size == 2
        assert g[0] is a and g[1] is b

    def test_rv_plus_group(self):
        a, b, c = (make_rv(seed=i, instrument=f'I{i}') for i in range(3))
        g = a + (b + c)
        assert g.size == 3 and g[0] is a and g[2] is c

    def test_group_plus_rv(self):
        a, b, c = (make_rv(seed=i, instrument=f'I{i}') for i in range(3))
        g = (a + b) + c
        assert g.size == 3 and g[2] is c

    def test_group_plus_group(self):
        a, b, c, d = (make_rv(seed=i, instrument=f'I{i}') for i in range(4))
        g = (a + b) + (c + d)
        assert g.size == 4 and [x for x in g] == [a, b, c, d]

    def test_addition_does_not_mutate(self):
        a, b, c = (make_rv(seed=i, instrument=f'I{i}') for i in range(3))
        g = a + b
        g2 = g + c
        assert g.size == 2 and g2.size == 3

    def test_sum_of_datasets(self):
        g = sum([make_rv(seed=i, instrument=f'I{i}') for i in range(3)])
        assert isinstance(g, RVDataGroup) and g.size == 3

    def test_sum_with_explicit_start(self):
        assert sum([], start=RVDataGroup()).size == 0

    def test_nested_group_is_flattened(self):
        a, b, c = (make_rv(seed=i, instrument=f'I{i}') for i in range(3))
        assert RVDataGroup([a, b + c]).size == 3

    def test_adding_a_number_raises(self):
        with pytest.raises(TypeError):
            make_rv() + 5
        with pytest.raises(TypeError):
            make_group() + 'x'

    def test_same_instance_twice_raises(self):
        a = make_rv()
        with pytest.raises(ValueError, match='cannot be added to a group twice'):
            a + a

    def test_non_rvdata_element_raises(self):
        with pytest.raises(TypeError, match='holds RVData objects'):
            RVDataGroup([make_rv(), 'not rv data'])


class TestCrossTypeAddition:
    """The shared container base must not let the two data types mix."""

    def make_lc(self):
        return LightCurveData(linspace(0, 1, 20), ones(20), passband='TESS')

    def test_rv_plus_light_curve_raises(self):
        with pytest.raises(TypeError):
            make_rv() + self.make_lc()

    def test_light_curve_plus_rv_raises(self):
        with pytest.raises(TypeError):
            self.make_lc() + make_rv()

    def test_rv_group_rejects_a_light_curve(self):
        """__add__ returns NotImplemented, so Python raises its own operand TypeError."""
        with pytest.raises(TypeError, match='unsupported operand'):
            make_group() + self.make_lc()

    def test_light_curve_group_rejects_rv_data(self):
        from pytransit.utils.io import LightCurveDataGroup
        with pytest.raises(TypeError, match='unsupported operand'):
            LightCurveDataGroup([self.make_lc()]) + make_rv()

    def test_constructing_a_mixed_group_gives_an_informative_error(self):
        with pytest.raises(TypeError, match='holds RVData objects'):
            RVDataGroup([make_rv(), self.make_lc()])

    def test_constructing_a_mixed_lc_group_gives_an_informative_error(self):
        from pytransit.utils.io import LightCurveDataGroup
        with pytest.raises(TypeError, match='holds LightCurveData objects'):
            LightCurveDataGroup([self.make_lc(), make_rv()])


class TestGroupProperties:
    def test_bulk_data(self):
        g = make_group()
        assert len(g.times) == len(g.rvs) == len(g.errors) == len(g.covariates) == 3
        for t, rv, e, cv in zip(g.times, g.rvs, g.errors, g.covariates):
            assert t.ndim == 1 and rv.shape == t.shape and e.shape == t.shape
            assert cv.ndim == 2 and cv.shape[0] == t.size

    def test_metadata(self):
        g = make_group()
        assert g.instruments == ['HARPS', 'HARPS-N', 'CARMENES']
        assert_array_equal(g.npts, [NPT] * 3)
        assert_array_equal(g.ncovs, [2] * 3)
        assert g.noises.dtype == float and g.noises.size == 3
        assert g.has_errors and g.has_covariates

    def test_time_range(self):
        g = make_group()
        assert_allclose(g.tmin, 0.0)
        assert_allclose(g.tmax, 500.0)

    def test_wnids_group_by_instrument(self):
        g = make_group()
        assert_array_equal(g.wnids, [0, 1, 2])
        g2 = RVDataGroup([make_rv(instrument='A', seed=0), make_rv(instrument='A', seed=1),
                          make_rv(instrument='B', seed=2)])
        assert_array_equal(g2.wnids, [0, 0, 1])

    def test_piis(self):
        g = RVDataGroup([make_rv(instrument='A', seed=0), make_rv(instrument='A', seed=1),
                         make_rv(instrument='B', seed=2)])
        assert_array_equal(g.piis, [0, 1, 0])

    def test_no_planet_index_properties(self):
        """pids and n_planets stayed out of the shared base."""
        g = make_group()
        assert not hasattr(g, 'pids')
        assert not hasattr(g, 'n_planets')
        assert not hasattr(make_rv(), 'pids')


class TestRvis:
    def test_returns_a_list_in_dataset_order(self):
        g = make_group()
        assert isinstance(g.rvis, list)          # an ndarray breaks RVLPF's `if rvis:`
        assert g.rvis == ['HARPS', 'HARPS-N', 'CARMENES']

    def test_duplicate_labels_raise(self):
        g = RVDataGroup([make_rv(instrument='HARPS', seed=0),
                         make_rv(instrument='HARPS-N', seed=1),
                         make_rv(instrument='HARPS', seed=2)])
        with pytest.raises(ValueError, match="Instrument labels \\['HARPS'\\] are used by more than one"):
            _ = g.rvis

    def test_duplicate_message_names_the_datasets(self):
        g = RVDataGroup([make_rv(instrument='A', seed=0), make_rv(instrument='A', seed=1)])
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            _ = g.rvis


class TestContainerProtocol:
    def test_integer_index(self):
        assert isinstance(make_group()[1], RVData)

    def test_slice_and_fancy_indexing(self):
        g = make_group()
        assert g[:2].size == 2
        assert g[[0, 2]][1] is g[2]
        assert g[array([True, False, True])].size == 2

    def test_wrong_length_boolean_index_raises(self):
        with pytest.raises(IndexError, match='Boolean index has 2 entries'):
            _ = make_group()[array([True, False])]

    def test_iteration_and_len(self):
        g = make_group()
        assert len(g) == 3 and [d for d in g] == g.data

    def test_repr(self):
        r = repr(make_group())
        assert '3 datasets' in r and '120 points' in r

    def test_select(self):
        g = make_group()
        assert g.select(instrument='HARPS').size == 1
        assert g.select(instrument='nope').size == 0

    def test_sorted_by(self):
        g = make_group()
        assert g.sorted_by('time')[0] is g[0]
        assert g.sorted_by('instrument')[0].instrument == 'CARMENES'
        assert g.sorted_by(lambda d: -d.time.min())[0] is g[2]

    def test_sorted_by_unknown_key_raises(self):
        with pytest.raises(ValueError, match='Unknown sort key'):
            make_group().sorted_by('passband')

    def test_empty_group(self):
        g = RVDataGroup()
        assert g.size == 0 and g.times == [] and g.rvis == []
        assert isnan(g.tmin) and isnan(g.tmax)
        assert not g.has_errors and not g.has_covariates


class TestRVLPFIntegration:
    def make_rv_dataset(self, instrument, t0, seed, k=30.0, offset=0.0):
        rng = default_rng(seed)
        t = linspace(t0, t0 + 80.0, 30)
        rv = offset + k * ones(30) * 0.0 + rng.normal(0.0, 2.0, 30)
        return RVData(t, rv, full(30, 2.0), instrument=instrument)

    def test_group_feeds_rvlpf(self):
        rvd = (self.make_rv_dataset('HARPS', 0.0, 0)
               + self.make_rv_dataset('HARPS-N', 200.0, 1))
        lpf = RVLPF('rv', nplanets=1, times=rvd.times, rvs=rvd.rvs, rves=rvd.errors,
                    rvis=rvd.rvis, is_transiting=[True])
        assert 'rv_shift_HARPS' in lpf.ps.names
        assert 'rv_err_HARPS-N' in lpf.ps.names
        assert 'rv_k_1' in lpf.ps.names and 't0_1' in lpf.ps.names
        assert isfinite(lpf.lnposterior(lpf.ps.mean_pv))

    def test_non_transiting_planet_uses_mean_anomaly(self):
        rvd = (self.make_rv_dataset('HARPS', 0.0, 0)
               + self.make_rv_dataset('CARMENES', 200.0, 1))
        lpf = RVLPF('rv2', nplanets=2, times=rvd.times, rvs=rvd.rvs, rves=rvd.errors,
                    rvis=rvd.rvis, is_transiting=[True, False])
        assert 't0_1' in lpf.ps.names and 'm0_2' in lpf.ps.names
        assert isfinite(lpf.lnposterior(lpf.ps.mean_pv))
