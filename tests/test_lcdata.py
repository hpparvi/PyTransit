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

import pickle

import matplotlib
import pytest

matplotlib.use('Agg')  # A headless backend, set before pytransit imports pyplot.

from matplotlib.colors import to_rgba
from matplotlib.pyplot import close
from numpy import (linspace, ones, zeros, full, array, arange, diff, nanstd, sqrt, nan, isnan,
                   isfinite, unique, float64, floor, concatenate, median)
from numpy.random import default_rng
from numpy.testing import assert_allclose, assert_array_equal

from pytransit import BaseLPF, RoadRunnerModel
from pytransit.utils.io import LCData, LCDataGroup

NPT = 100


def make_lc(npt=NPT, t0=0.0, ncov=2, passband='TESS', instrument='TESS', sector=1, segment=0, seed=0,
            pids=None, error=None):
    rng = default_rng(seed)
    t = linspace(t0 + 0.9, t0 + 1.1, npt)
    f = 1.0 + rng.normal(0.0, 1e-3, npt)
    cv = rng.normal(0.0, 1.0, (npt, ncov)) if ncov else None
    return LCData(t, f, error=error, covariates=cv, passband=passband, instrument=instrument,
                          sector=sector, segment=segment, pids=pids)


def make_group():
    """Four light curves with passbands TESS, g, g, r across two instruments."""
    return LCDataGroup([
        make_lc(passband='TESS', instrument='TESS', sector=1, seed=0),
        make_lc(passband='g', instrument='MuSCAT2', sector=-1, t0=1.0, seed=1),
        make_lc(passband='g', instrument='MuSCAT2', sector=-1, t0=2.0, seed=2, segment=1),
        make_lc(passband='r', instrument='MuSCAT2', sector=-1, t0=2.0, seed=3, segment=1)])


class TestCoercion:
    def test_lists_become_float64_arrays(self):
        lc = LCData([1.0, 2.0, 3.0], [1, 1, 1])
        assert lc.time.dtype == float64 and lc.flux.dtype == float64
        assert lc.time.ndim == 1 and lc.flux.ndim == 1

    def test_int_arrays_become_float(self):
        lc = LCData(arange(10), ones(10, int))
        assert lc.time.dtype == float64 and lc.flux.dtype == float64

    def test_sizes(self):
        lc = make_lc(ncov=3)
        assert lc.size == NPT and lc.npt == NPT and lc.ncov == 3

    def test_1d_covariates_become_a_single_column(self):
        lc = LCData(arange(10), ones(10), covariates=arange(10))
        assert lc.covariates.shape == (10, 1)

    def test_arrays_are_not_copied(self):
        t, f = linspace(0, 1, 10), ones(10)
        lc = LCData(t, f)
        assert lc.time is t and lc.flux is f


class TestValidation:
    def test_flux_length_mismatch(self):
        with pytest.raises(ValueError, match="'flux' has 9 points but 'time' has 10"):
            LCData(arange(10), ones(9))

    def test_covariate_row_mismatch(self):
        with pytest.raises(ValueError, match="'covariates' has 9 rows"):
            LCData(arange(10), ones(10), covariates=zeros((9, 2)))

    def test_3d_covariates(self):
        with pytest.raises(ValueError, match="'covariates' must be a 1D or 2D array"):
            LCData(arange(10), ones(10), covariates=zeros((10, 2, 2)))

    def test_2d_time(self):
        with pytest.raises(ValueError, match="'time' must be a 1D array"):
            LCData(zeros((10, 2)), ones(20))

    def test_nonfinite_time(self):
        t = linspace(0, 1, 10)
        t[3] = nan
        with pytest.raises(ValueError, match="'time' contains non-finite values"):
            LCData(t, ones(10))

    def test_empty_passband(self):
        with pytest.raises(ValueError, match="'passband' cannot be empty"):
            LCData(arange(10), ones(10), passband=[])

    def test_zero_nsamples(self):
        with pytest.raises(ValueError, match="'nsamples' must be at least one"):
            LCData(arange(10), ones(10), nsamples=0)

    def test_negative_exptime(self):
        with pytest.raises(ValueError, match="'exptime' must be a finite non-negative"):
            LCData(arange(10), ones(10), exptime=-1.0)

    def test_nonintegral_sector(self):
        with pytest.raises(ValueError, match="'sector' must be an integer"):
            LCData(arange(10), ones(10), sector=1.5)

    def test_nonpositive_noise(self):
        with pytest.raises(ValueError, match="'noise' must be a finite positive"):
            LCData(arange(10), ones(10), noise=0.0)

    def test_nonfinite_flux_warns(self):
        f = ones(10)
        f[2] = nan
        with pytest.warns(UserWarning, match="'flux' contains non-finite values"):
            LCData(arange(10), f)

    def test_supersampling_without_exptime_warns(self):
        with pytest.warns(UserWarning, match="Supersampling has no effect"):
            LCData(arange(10), ones(10), nsamples=5, exptime=0.0)


class TestDefaults:
    def test_covariates_default_to_empty_matrix(self):
        lc = LCData(arange(10), ones(10))
        assert lc.covariates.shape == (10, 0) and lc.ncov == 0

    def test_noise_is_estimated_from_the_flux(self):
        rng = default_rng(0)
        f = 1.0 + rng.normal(0, 1e-3, NPT)
        lc = LCData(linspace(0, 1, NPT), f)
        assert_allclose(lc.noise, nanstd(diff(f)) / sqrt(2))

    def test_explicit_noise_is_kept(self):
        lc = LCData(arange(10), ones(10), noise=1e-3)
        assert lc.noise == 1e-3

    def test_single_point_noise_is_nan(self):
        lc = LCData([1.0], [1.0])
        assert isnan(lc.noise)

    def test_string_passband_becomes_tuple(self):
        assert LCData(arange(10), ones(10), passband='TESS').passband == ('TESS',)

    def test_sequence_passband_kept_in_order(self):
        lc = LCData(arange(10), ones(10), passband=['g', 'r', 'i', 'z'])
        assert lc.passband == ('g', 'r', 'i', 'z')

    def test_error_repeats_the_noise(self):
        lc = LCData(arange(10), ones(10), noise=2e-3)
        assert lc.error.shape == (10,)
        assert_allclose(lc.error, 2e-3)


class TestError:
    def test_error_defaults_to_the_repeated_noise(self):
        lc = LCData(arange(10), ones(10), noise=2e-3)
        assert not lc.has_error
        assert_allclose(lc.error, 2e-3)

    def test_given_error_is_used(self):
        e = linspace(1e-3, 2e-3, 10)
        lc = LCData(arange(10), ones(10), error=e)
        assert lc.has_error
        assert_allclose(lc.error, e)

    def test_error_is_coerced_to_float64(self):
        lc = LCData(arange(10), ones(10), error=[1] * 10)
        assert lc.error.dtype == float64 and lc.error.shape == (10,)

    def test_error_is_independent_of_noise(self):
        """noise stays the point-to-point estimate; it is not derived from error."""
        rng = default_rng(0)
        f = 1.0 + rng.normal(0, 1e-3, NPT)
        lc = LCData(linspace(0, 1, NPT), f, error=full(NPT, 5e-3))
        assert_allclose(lc.noise, nanstd(diff(f)) / sqrt(2))
        assert_allclose(lc.error, 5e-3)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="'error' has 9 points but 'time' has 10"):
            LCData(arange(10), ones(10), error=ones(9))

    def test_2d_error_raises(self):
        with pytest.raises(ValueError, match="'error' must be a 1D array"):
            LCData(arange(10), ones(10), error=ones((10, 2)))

    def test_nonpositive_error_raises(self):
        e = ones(10)
        e[4] = 0.0
        with pytest.raises(ValueError, match="'error' contains non-positive values"):
            LCData(arange(10), ones(10), error=e)

    def test_nonfinite_error_warns_but_is_kept(self):
        e = ones(10)
        e[4] = nan
        with pytest.warns(UserWarning, match="'error' contains non-finite values"):
            lc = LCData(arange(10), ones(10), error=e)
        assert isnan(lc.error[4])

    def test_group_errors_and_has_errors(self):
        a = make_lc(seed=0)
        b = LCData(linspace(2, 3, NPT), ones(NPT), error=full(NPT, 1e-3))
        g = a + b
        assert not g.has_errors                     # only one of the two has errors
        assert_allclose(g.errors[0], a.noise)
        assert_allclose(g.errors[1], 1e-3)
        assert LCDataGroup([b]).has_errors

    def test_group_errors_feed_baselpf(self):
        lcs = LCDataGroup([
            LCData(linspace(0.9, 1.1, NPT), ones(NPT), error=full(NPT, 1e-3)),
            LCData(linspace(1.9, 2.1, NPT), ones(NPT), error=full(NPT, 2e-3))])
        lpf = BaseLPF('err', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                      errors=lcs.errors if lcs.has_errors else None, pbids=lcs.pbids,
                      wnids=lcs.wnids, tm=RoadRunnerModel('quadratic'))
        assert_allclose(lpf.errora[:NPT], 1e-3)
        assert_allclose(lpf.errora[NPT:], 2e-3)


class TestPids:
    def test_default_is_none(self):
        lc = LCData(arange(10), ones(10))
        assert lc.pids is None and not lc.has_pids

    def test_single_int_becomes_a_tuple(self):
        lc = LCData(arange(10), ones(10), pids=0)
        assert lc.pids == (0,) and lc.has_pids

    def test_sequence_is_kept_in_order(self):
        lc = LCData(arange(10), ones(10), pids=[2, 0, 1])
        assert lc.pids == (2, 0, 1)

    def test_empty_sequence_differs_from_none(self):
        """An empty tuple means 'no planet transits here'; None means 'not specified'."""
        lc = LCData(arange(10), ones(10), pids=[])
        assert lc.pids == () and lc.has_pids

    def test_numpy_ints_are_accepted(self):
        lc = LCData(arange(10), ones(10), pids=array([0, 2]))
        assert lc.pids == (0, 2)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match='cannot contain negative planet indices'):
            LCData(arange(10), ones(10), pids=[0, -1])

    def test_duplicate_index_raises(self):
        with pytest.raises(ValueError, match='duplicate planet indices'):
            LCData(arange(10), ones(10), pids=[0, 1, 0])

    def test_nonintegral_index_raises(self):
        with pytest.raises(ValueError, match="'pids' must be an integer"):
            LCData(arange(10), ones(10), pids=[0, 1.5])

    def test_non_sequence_raises(self):
        with pytest.raises(ValueError, match="'pids' must be an integer or a sequence"):
            LCData(arange(10), ones(10), pids='0')

    def test_repr_shows_pids(self):
        assert 'pids=(0, 1)' in repr(LCData(arange(10), ones(10), pids=[0, 1]))

    def test_group_pids(self):
        g = LCDataGroup([make_lc(seed=0), make_lc(seed=1, t0=1.0)])
        assert g.pids == [None, None]
        assert not g.has_pids and g.n_planets == 0

    def test_group_has_pids_requires_all(self):
        a = LCData(linspace(0, 1, 10), ones(10), pids=[0])
        b = LCData(linspace(2, 3, 10), ones(10))
        assert not (a + b).has_pids
        assert LCDataGroup([a]).has_pids

    def test_group_n_planets(self):
        a = LCData(linspace(0, 1, 10), ones(10), pids=[0, 2])
        b = LCData(linspace(2, 3, 10), ones(10), pids=[1])
        g = a + b
        assert g.pids == [(0, 2), (1,)]
        assert g.has_pids and g.n_planets == 3

    def test_group_n_planets_ignores_unspecified(self):
        a = LCData(linspace(0, 1, 10), ones(10), pids=[0, 1])
        b = LCData(linspace(2, 3, 10), ones(10))
        assert (a + b).n_planets == 2


class TestAddition:
    def test_lc_plus_lc(self):
        a, b = make_lc(seed=0), make_lc(seed=1)
        g = a + b
        assert isinstance(g, LCDataGroup) and g.size == 2
        assert g[0] is a and g[1] is b

    def test_lc_plus_group(self):
        a, b, c = make_lc(seed=0), make_lc(seed=1), make_lc(seed=2)
        g = a + (b + c)
        assert g.size == 3 and g[0] is a and g[2] is c

    def test_group_plus_lc(self):
        a, b, c = make_lc(seed=0), make_lc(seed=1), make_lc(seed=2)
        g = (a + b) + c
        assert g.size == 3 and g[2] is c

    def test_group_plus_group(self):
        a, b, c, d = (make_lc(seed=i) for i in range(4))
        g = (a + b) + (c + d)
        assert g.size == 4 and [x for x in g] == [a, b, c, d]

    def test_addition_does_not_mutate(self):
        a, b, c = make_lc(seed=0), make_lc(seed=1), make_lc(seed=2)
        g = a + b
        g2 = g + c
        assert g.size == 2 and g2.size == 3

    def test_sum_of_light_curves(self):
        lcs = [make_lc(seed=i) for i in range(3)]
        g = sum(lcs)
        assert isinstance(g, LCDataGroup) and g.size == 3

    def test_sum_with_explicit_start(self):
        assert sum([], start=LCDataGroup()).size == 0

    def test_nested_group_is_flattened(self):
        a, b, c = make_lc(seed=0), make_lc(seed=1), make_lc(seed=2)
        assert LCDataGroup([a, b + c]).size == 3

    def test_adding_a_number_raises(self):
        with pytest.raises(TypeError):
            make_lc() + 5
        with pytest.raises(TypeError):
            (make_lc(seed=0) + make_lc(seed=1)) + 'x'

    def test_same_instance_twice_raises(self):
        a = make_lc()
        with pytest.raises(ValueError, match='cannot be added to a group twice'):
            a + a

    def test_non_lightcurve_element_raises(self):
        with pytest.raises(TypeError, match='holds LCData objects'):
            LCDataGroup([make_lc(), 'not a light curve'])


class TestGroupProperties:
    def test_bulk_data_shapes(self):
        g = make_group()
        assert len(g.times) == len(g.fluxes) == len(g.covariates) == len(g.errors) == 4
        for t, f, cv, e in zip(g.times, g.fluxes, g.covariates, g.errors):
            assert t.ndim == 1 and t.dtype == float64
            assert f.shape == t.shape
            assert cv.ndim == 2 and cv.shape[0] == t.size
            assert e.shape == t.shape

    def test_errors_repeat_the_noises(self):
        g = make_group()
        for e, n in zip(g.errors, g.noises):
            assert_allclose(e, n)

    def test_metadata_arrays(self):
        g = make_group()
        assert g.instruments == ['TESS', 'MuSCAT2', 'MuSCAT2', 'MuSCAT2']
        assert_array_equal(g.sectors, [1, -1, -1, -1])
        assert_array_equal(g.segments, [0, 0, 1, 1])
        assert g.sectors.dtype == int and g.nsamples.dtype == int
        assert g.exptimes.dtype == float and g.noises.dtype == float

    def test_sizes(self):
        g = make_group()
        assert g.size == 4 and len(g) == 4
        assert_array_equal(g.npts, [NPT] * 4)
        assert_array_equal(g.ncovs, [2] * 4)
        assert g.has_covariates

    def test_has_covariates_false_without_them(self):
        g = LCDataGroup([make_lc(ncov=0), make_lc(ncov=0, seed=1)])
        assert not g.has_covariates

    def test_time_range(self):
        g = make_group()
        assert_allclose(g.tmin, 0.9)
        assert_allclose(g.tmax, 3.1)

    def test_passband_names_ordered_by_first_appearance(self):
        assert make_group().passband_names == ['TESS', 'g', 'r']

    def test_pbids(self):
        g = make_group()
        assert_array_equal(g.pbids, [0, 1, 1, 2])
        assert g.pbids.dtype == int

    def test_ins_and_piis(self):
        g = make_group()
        assert g.ins == g.instruments
        assert_array_equal(g.piis, [0, 0, 1, 2])


class TestLCSlices:
    def test_slices_split_a_concatenated_array(self):
        g = make_group()
        timea = concatenate(g.times)
        for sl, t in zip(g.lcslices, g.times):
            assert_array_equal(timea[sl], t)

    def test_slices_are_slice_objects_covering_the_array(self):
        g = make_group()
        assert all(isinstance(sl, slice) for sl in g.lcslices)
        assert len(g.lcslices) == g.size
        assert g.lcslices[0].start == 0
        assert g.lcslices[-1].stop == int(g.npts.sum())

    def test_slices_follow_uneven_light_curve_lengths(self):
        g = LCDataGroup([make_lc(npt=10, seed=0), make_lc(npt=25, t0=1.0, seed=1),
                         make_lc(npt=5, t0=2.0, seed=2)])
        assert g.lcslices == [slice(0, 10), slice(10, 35), slice(35, 40)]

    def test_empty_group_has_no_slices(self):
        assert LCDataGroup().lcslices == []


class TestMultiPassband:
    def test_composite_passband_is_stored(self):
        lc = make_lc(passband=['g', 'r', 'i', 'z'])
        assert lc.passband == ('g', 'r', 'i', 'z')

    def test_pbids_raises_naming_the_offender(self):
        g = LCDataGroup([make_lc(passband='TESS'),
                                 make_lc(passband=['g', 'r', 'i', 'z'], seed=1)])
        with pytest.raises(ValueError, match=r'Light curves \[1\] have more than one passband'):
            _ = g.pbids

    def test_passband_names_still_lists_components(self):
        g = LCDataGroup([make_lc(passband='TESS'),
                                 make_lc(passband=['g', 'r'], seed=1)])
        assert g.passband_names == ['TESS', 'g', 'r']


class TestNoiseIds:
    def test_default_metadata_gives_one_block(self):
        g = LCDataGroup([make_lc(instrument='', sector=-1, segment=0, seed=i)
                                 for i in range(3)])
        assert_array_equal(g.wnids, [0, 0, 0])

    def test_grouping_follows_metadata(self):
        g = make_group()
        assert_array_equal(g.wnids, [0, 1, 2, 2])

    def test_ids_are_zero_based_and_contiguous(self):
        g = make_group()
        w = g.wnids
        assert w.min() == 0
        assert set(w.tolist()) == set(range(w.max() + 1))

    def test_equal_metadata_gives_equal_id(self):
        g = LCDataGroup([make_lc(instrument='A', sector=2, seed=0),
                                 make_lc(instrument='A', sector=2, seed=1),
                                 make_lc(instrument='B', sector=2, seed=2)])
        assert g.wnids[0] == g.wnids[1] != g.wnids[2]


class TestContainerProtocol:
    def test_integer_index_gives_a_light_curve(self):
        assert isinstance(make_group()[1], LCData)

    def test_slice_gives_a_group(self):
        g = make_group()[:2]
        assert isinstance(g, LCDataGroup) and g.size == 2

    def test_integer_list_index(self):
        g = make_group()
        sub = g[[0, 2]]
        assert sub.size == 2 and sub[1] is g[2]

    def test_boolean_array_index(self):
        g = make_group()
        sub = g[array([True, False, True, False])]
        assert sub.size == 2 and sub[0] is g[0]

    def test_wrong_length_boolean_index_raises(self):
        with pytest.raises(IndexError, match='Boolean index has 2 entries'):
            _ = make_group()[array([True, False])]

    def test_iteration_order(self):
        g = make_group()
        assert [d for d in g] == g.data

    def test_repr_reports_size(self):
        r = repr(make_group())
        assert '4 light curves' in r and '400 points' in r

    def test_select(self):
        g = make_group()
        assert g.select(instrument='MuSCAT2').size == 3
        assert g.select(passband='g').size == 2
        assert g.select(instrument='MuSCAT2', segment=1).size == 2

    def test_select_with_a_sequence_of_values(self):
        g = make_group()
        assert g.select(passband=['g', 'r']).size == 3
        assert g.select(instrument=['TESS', 'MuSCAT2']).size == 4
        assert g.select(sector=array([1, 5])).size == 1
        assert g.select(passband=('g', 'r'), instrument='MuSCAT2').size == 3

    def test_select_with_an_empty_sequence_selects_nothing(self):
        assert make_group().select(passband=[]).size == 0

    def test_select_pids_with_a_sequence(self):
        g = LCDataGroup([make_lc(pids=(0, 1), seed=0), make_lc(pids=2, t0=1.0, seed=1),
                         make_lc(t0=2.0, seed=2)])
        assert g.select(pids=[0]).size == 1
        assert g.select(pids=[0, 2]).size == 2
        assert g.select(pids=[3]).size == 0

    def test_sorted_by_time(self):
        g = make_group()
        assert g.sorted_by('time')[0] is g[0]
        assert g.sorted_by(lambda d: -d.time.min())[0] is g[2]

    def test_sorted_by_unknown_key_raises(self):
        with pytest.raises(ValueError, match='Unknown sort key'):
            make_group().sorted_by('nonsense')


class TestEmptyGroup:
    def test_empty_group_is_legal(self):
        g = LCDataGroup()
        assert g.size == 0 and len(g) == 0
        assert g.times == [] and g.passband_names == []
        assert g.npts.size == 0 and g.wnids.size == 0
        assert isnan(g.tmin) and isnan(g.tmax)
        assert not g.has_covariates


class TestLPFIntegration:
    def test_group_slices_match_the_lpf_slices(self):
        lcs = (make_lc(npt=10, seed=0) + make_lc(npt=25, t0=1.0, seed=1) + make_lc(npt=5, t0=2.0, seed=2))
        lpf = BaseLPF('test', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                      pbids=lcs.pbids, tm=RoadRunnerModel('quadratic'))
        assert lcs.lcslices == lpf.lcslices

    def test_group_feeds_baselpf(self):
        lcs = (make_lc(passband='TESS', instrument='TESS', sector=1, seed=0)
               + make_lc(passband='g', instrument='MuSCAT2', t0=1.0, seed=1)
               + make_lc(passband='g', instrument='MuSCAT2', t0=2.0, seed=2))
        lpf = BaseLPF('test', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                      pbids=lcs.pbids, covariates=lcs.covariates, wnids=lcs.wnids,
                      nsamples=lcs.nsamples, exptimes=lcs.exptimes,
                      tm=RoadRunnerModel('quadratic'))
        assert lpf.nlc == 3 and lpf.npb == 2
        assert_array_equal(lpf.pbids, lcs.pbids)
        assert lpf.n_noise_blocks == unique(lcs.wnids).size
        assert isfinite(lpf.lnposterior(lpf.ps.mean_pv))

    def test_group_without_covariates_feeds_baselpf(self):
        lcs = (make_lc(ncov=0, seed=0) + make_lc(ncov=0, t0=1.0, seed=1))
        lpf = BaseLPF('nocov', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                      pbids=lcs.pbids, covariates=lcs.covariates if lcs.has_covariates else None,
                      wnids=lcs.wnids, tm=RoadRunnerModel('quadratic'))
        assert lpf.nlc == 2
        assert isfinite(lpf.lnposterior(lpf.ps.mean_pv))

    def test_linear_model_baseline_recipe(self):
        from pytransit import LinearModelBaseline

        lcs = (make_lc(passband='TESS', instrument='TESS', seed=0)
               + make_lc(passband='g', instrument='MuSCAT2', t0=1.0, seed=1))
        lpf = BaseLPF('lm', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                      pbids=lcs.pbids, covariates=lcs.covariates, wnids=lcs.wnids,
                      tm=RoadRunnerModel('quadratic'))
        lpf.ins, lpf.piis = lcs.ins, lcs.piis
        LinearModelBaseline(lpf)
        names = [p.name for p in lpf.ps]
        assert any('MuSCAT2' in n for n in names)
        assert any('TESS' in n for n in names)


def make_spiked_lc(npt=NPT, spikes=(42,), amplitude=0.02, seed=0, **kwargs):
    """A clean light curve with a large flux spike at each of the given indices."""
    rng = default_rng(seed)
    f = 1.0 + rng.normal(0.0, 1e-3, npt)
    for i in spikes:
        f[i] += amplitude
    return LCData(linspace(0.9, 1.1, npt), f, covariates=arange(float(npt)).reshape(npt, 1), **kwargs)


class TestRunningMedian:
    def test_running_median_has_the_right_length_and_tracks_the_flux(self):
        lc = make_lc()
        m = lc.running_median(15)
        assert m.size == lc.size
        assert_allclose(m, 1.0, atol=1e-3)

    def test_the_edges_are_not_biased_low(self):
        """medfilt zero-pads, which drags the edge medians below the data."""
        from scipy.signal import medfilt
        rng = default_rng(0)
        f = 1.0 + rng.normal(0.0, 1e-3, NPT)
        lc = LCData(linspace(0.9, 1.1, NPT), f)
        m, raw = lc.running_median(15), medfilt(f, 15)
        assert_allclose(m[:7], median(f[:15]))
        assert_allclose(m[-7:], median(f[-15:]))
        assert raw[0] < m[0] and raw[-1] < m[-1]             # The unrepaired filter is biased low.
        assert abs(m[0] - 1.0) < abs(raw[0] - 1.0)

    def test_a_width_wider_than_the_light_curve_stays_finite(self):
        lc = make_lc(npt=9)
        assert isfinite(lc.running_median(15)).all()

    def test_an_even_width_raises(self):
        with pytest.raises(ValueError, match="'width' must be a positive odd integer"):
            make_lc().running_median(10)

    def test_a_non_positive_width_raises(self):
        with pytest.raises(ValueError, match="'width' must be a positive odd integer"):
            make_lc().running_median(-1)

    def test_a_non_integral_width_raises(self):
        with pytest.raises(ValueError, match="'width' must be an integer"):
            make_lc().running_median(15.5)


class TestOutlierRemoval:
    def test_outlier_mask_flags_the_spike(self):
        lc = make_spiked_lc(spikes=(42,))
        assert_array_equal(lc.outlier_mask(5.0, 15).nonzero()[0], array([42]))

    def test_remove_outliers_drops_the_spike_and_returns_the_count(self):
        lc = make_spiked_lc(spikes=(42,))
        assert lc.remove_outliers(5.0, 15) == 1
        assert lc.npt == NPT - 1

    def test_every_per_point_array_is_filtered_consistently(self):
        lc = make_spiked_lc(spikes=(42,), error=full(NPT, 1e-3))
        lc.remove_outliers(5.0, 15)
        assert lc.time.size == lc.flux.size == lc.covariates.shape[0] == lc.error.size == NPT - 1
        # The covariates carry their point index, so the survivors must skip exactly 42.
        assert_array_equal(lc.covariates[40:44, 0], array([40.0, 41.0, 43.0, 44.0]))

    def test_several_spikes_are_caught_in_one_pass(self):
        """The robust scatter is what makes this work: a plain std would be inflated by them."""
        lc = make_spiked_lc(spikes=(10, 42, 77), amplitude=0.05)
        assert lc.remove_outliers(5.0, 15) == 3

    def test_a_clean_light_curve_keeps_its_points(self):
        lc = make_lc()
        assert lc.remove_outliers(5.0, 15) == 0
        assert lc.npt == NPT

    def test_an_estimated_noise_is_refreshed(self):
        lc = make_spiked_lc(spikes=(42,))
        assert lc.remove_outliers(5.0, 15) == 1
        assert_allclose(lc.noise, nanstd(diff(lc.flux)) / sqrt(2))

    def test_an_explicit_noise_is_left_alone(self):
        lc = make_spiked_lc(spikes=(42,), noise=5e-3)
        lc.remove_outliers(5.0, 15)
        assert lc.noise == 5e-3

    def test_a_constant_light_curve_removes_nothing(self):
        """A zero scatter must not divide by zero."""
        lc = LCData(linspace(0.9, 1.1, NPT), ones(NPT))
        assert not lc.outlier_mask(3.0, 15).any()
        assert lc.remove_outliers(3.0, 15) == 0

    def test_an_empty_light_curve_is_handled(self):
        lc = LCData([], [])
        assert lc.outlier_mask(3.0, 15).size == 0
        assert lc.remove_outliers(3.0, 15) == 0

    def test_group_removal_returns_the_total_and_keeps_the_light_curves(self):
        g = LCDataGroup([make_spiked_lc(spikes=(42,), seed=0),
                         make_spiked_lc(spikes=(10, 60), seed=1)])
        assert g.remove_outliers(5.0, 15) == 3
        assert g.size == 2
        assert_array_equal(g.npts, array([NPT - 1, NPT - 2]))

    def test_group_slices_follow_a_removal(self):
        g = LCDataGroup([make_spiked_lc(spikes=(42,), seed=0),
                         make_spiked_lc(spikes=(10, 60), seed=1)])
        g.remove_outliers(5.0, 15)
        assert g.lcslices == [slice(0, NPT - 1), slice(NPT - 1, 2 * NPT - 3)]


class TestTimeCovariates:
    def test_time_covariates_are_added(self):
        lc = make_lc(ncov=0)
        lc.add_time_covariates(3)
        assert lc.ncov == 3

    def test_the_first_column_is_the_time_normalised_to_pm_one(self):
        lc = make_lc(ncov=0)
        lc.add_time_covariates(1)
        tn = lc.covariates[:, 0]
        assert_allclose([tn.min(), tn.max()], [-1.0, 1.0])
        assert_allclose(tn, 2 * (lc.time - lc.time.min()) / (lc.time.max() - lc.time.min()) - 1)

    def test_the_columns_are_the_powers_of_the_normalised_time(self):
        lc = make_lc(ncov=0)
        lc.add_time_covariates(4)
        tn = lc.covariates[:, 0]
        for i in range(1, 4):
            assert_allclose(lc.covariates[:, i], tn ** (i + 1))

    def test_the_columns_are_concatenated_with_the_existing_covariates(self):
        lc = make_lc(ncov=2)
        original = lc.covariates.copy()
        lc.add_time_covariates(2)
        assert lc.ncov == 4
        assert_array_equal(lc.covariates[:, :2], original)

    def test_repeated_calls_append(self):
        lc = make_lc(ncov=2)
        lc.add_time_covariates(1)
        lc.add_time_covariates(1)
        assert lc.ncov == 4
        assert_array_equal(lc.covariates[:, 2], lc.covariates[:, 3])

    def test_a_constant_time_gives_a_zero_column(self):
        lc = LCData(full(5, 2.0), ones(5))
        lc.add_time_covariates(1)
        assert_array_equal(lc.covariates[:, 0], zeros(5))

    def test_an_invalid_order_raises(self):
        with pytest.raises(ValueError, match="'order' must be a positive integer"):
            make_lc().add_time_covariates(0)
        with pytest.raises(ValueError, match="'order' must be an integer"):
            make_lc().add_time_covariates(1.5)

    def test_the_group_adds_them_to_every_light_curve(self):
        g = make_group()
        g.add_time_covariates(2)
        assert_array_equal(g.ncovs, array([4, 4, 4, 4]))
        for lc in g:                        # Each light curve is normalised on its own time.
            assert_allclose([lc.covariates[:, 2].min(), lc.covariates[:, 2].max()], [-1.0, 1.0])

    def test_the_columns_are_filtered_by_an_outlier_removal(self):
        lc = make_spiked_lc(spikes=(42,))
        lc.add_time_covariates(1)
        lc.remove_outliers(5.0, 15)
        assert lc.covariates.shape == (NPT - 1, 2)
        assert_array_equal(lc.covariates[40:44, 0], array([40.0, 41.0, 43.0, 44.0]))


class TestLinearModel:
    def test_a_flux_linear_in_the_covariates_is_reproduced(self):
        rng = default_rng(0)
        cv = rng.normal(0.0, 1.0, (NPT, 2))
        f = 1.0 + 0.01 * cv[:, 0] - 0.005 * cv[:, 1]
        lc = LCData(linspace(0.9, 1.1, NPT), f, covariates=cv)
        assert_allclose(lc.linear_model(), f, atol=1e-12)

    def test_the_model_is_insensitive_to_the_covariate_scaling(self):
        """The columns are standardised, so a badly scaled covariate gives the same fit."""
        rng = default_rng(0)
        cv = rng.normal(0.0, 1.0, (NPT, 2))
        f = 1.0 + 0.01 * cv[:, 0] - 0.005 * cv[:, 1]
        t = linspace(0.9, 1.1, NPT)
        scaled = cv.copy()
        scaled[:, 0] = 2.46e6 + 1e4 * scaled[:, 0]
        assert_allclose(LCData(t, f, covariates=cv).linear_model(),
                        LCData(t, f, covariates=scaled).linear_model(), atol=1e-10)

    def test_a_constant_covariate_column_is_handled(self):
        cv = ones((NPT, 1))
        lc = LCData(linspace(0.9, 1.1, NPT), 1.0 + default_rng(0).normal(0, 1e-3, NPT), covariates=cv)
        assert_allclose(lc.linear_model(), lc.flux.mean(), atol=1e-12)

    def test_without_covariates_the_model_is_the_mean_flux(self):
        lc = make_lc(ncov=0)
        assert_allclose(lc.linear_model(), lc.flux.mean())

    def test_the_model_has_one_value_per_point(self):
        assert make_lc().linear_model().size == NPT


class TestMarking:
    def test_light_curves_are_unmarked_by_default(self):
        g = make_group()
        assert not g[0].marked
        assert_array_equal(g.marked, zeros(4, bool))
        assert g.n_marked == 0

    def test_mark_for_removal_marks_the_given_indices(self):
        g = make_group()
        assert g.mark_for_removal([0, 2]) is None
        assert_array_equal(g.marked, array([True, False, True, False]))
        assert g.n_marked == 2

    def test_mark_for_removal_accepts_a_single_index(self):
        g = make_group()
        g.mark_for_removal(1)
        assert g.n_marked == 1 and g[1].marked

    def test_mark_for_removal_accepts_a_set(self):
        g = make_group()
        g.mark_for_removal({1, 3})
        assert_array_equal(g.marked, array([False, True, False, True]))

    def test_mark_for_removal_accepts_a_negative_index(self):
        g = make_group()
        g.mark_for_removal(-1)
        assert g[3].marked and g.n_marked == 1

    def test_mark_for_removal_accepts_a_boolean_mask(self):
        g = make_group()
        g.mark_for_removal(g.segments == 1)
        assert_array_equal(g.marked, array([False, False, True, True]))

    def test_mark_for_removal_accepts_a_slice(self):
        g = make_group()
        g.mark_for_removal(slice(1, 3))
        assert g.n_marked == 2 and g[1].marked and g[2].marked

    def test_mark_for_removal_is_additive_and_idempotent(self):
        g = make_group()
        g.mark_for_removal(0)
        g.mark_for_removal([2, 2])
        assert_array_equal(g.marked, array([True, False, True, False]))

    def test_mark_for_removal_out_of_range_raises(self):
        with pytest.raises(IndexError, match='out of range'):
            make_group().mark_for_removal(4)
        with pytest.raises(IndexError, match='out of range'):
            make_group().mark_for_removal(-5)

    def test_mark_for_removal_rejects_a_boolean_scalar(self):
        with pytest.raises(ValueError, match='got a boolean'):
            make_group().mark_for_removal(True)

    def test_mark_for_removal_rejects_a_non_integral_index(self):
        with pytest.raises(ValueError, match='must be an integer'):
            make_group().mark_for_removal(1.5)

    def test_mark_for_removal_wrong_length_mask_raises(self):
        with pytest.raises(IndexError, match='Boolean index has 2 entries'):
            make_group().mark_for_removal(array([True, False]))

    def test_unmark_clears_the_given_indices(self):
        g = make_group()
        g.mark_for_removal([0, 2])
        g.unmark(0)
        assert_array_equal(g.marked, array([False, False, True, False]))

    def test_unmark_without_arguments_clears_everything(self):
        g = make_group()
        g.mark_for_removal([0, 2])
        g.unmark()
        assert g.n_marked == 0

    def test_marks_are_shared_with_a_selection(self):
        g = make_group()
        g.select(instrument='MuSCAT2').mark_for_removal(0)
        assert g[1].marked and g.n_marked == 1

    def test_marks_are_shared_with_a_slice(self):
        g = make_group()
        g[1:].mark_for_removal(0)
        assert g[1].marked and g.n_marked == 1

    def test_marked_light_curves_can_be_selected(self):
        g = make_group()
        g.mark_for_removal([1, 3])
        assert g.select(marked=True).size == 2

    def test_marking_does_not_change_the_bulk_data(self):
        g = make_group()
        g.mark_for_removal(0)
        assert g.size == 4 and len(g.times) == 4 and g.pbids.size == 4

    def test_reprs_report_the_marks_only_when_set(self):
        g = make_group()
        assert 'marked' not in repr(g) and 'marked' not in repr(g[0])
        g.mark_for_removal([0, 2])
        assert '2 marked for removal' in repr(g) and 'marked' in repr(g[0])

    def test_marks_survive_pickling(self):
        g = make_group()
        g.mark_for_removal(1)
        assert_array_equal(pickle.loads(pickle.dumps(g)).marked, g.marked)

    def test_a_light_curve_without_the_attribute_reads_as_unmarked(self):
        """A light curve pickled before `marked` existed must not raise."""
        lc = make_lc()
        del lc.__dict__['marked']
        assert lc.marked is False


class TestRemoveMarked:
    def test_remove_marked_removes_in_place(self):
        g = make_group()
        keep = [g[1], g[3]]
        g.mark_for_removal([0, 2])
        assert g.remove_marked() is None
        assert g.size == 2 and g[0] is keep[0] and g[1] is keep[1]

    def test_remove_marked_without_marks_is_a_no_op(self):
        g = make_group()
        g.remove_marked()
        assert g.size == 4

    def test_survivors_are_unmarked(self):
        g = make_group()
        g.mark_for_removal(0)
        g.remove_marked()
        assert g.n_marked == 0

    def test_removed_light_curves_keep_their_mark(self):
        g = make_group()
        lc = g[0]
        g.mark_for_removal(0)
        g.remove_marked()
        assert lc.marked

    def test_removing_everything_leaves_a_legal_empty_group(self):
        g = make_group()
        g.mark_for_removal(slice(None))
        g.remove_marked()
        assert g.size == 0 and g.times == [] and g.passband_names == [] and g.lcslices == []

    def test_derived_ids_are_renumbered(self):
        g = make_group()
        g.mark_for_removal(0)                        # The only TESS light curve.
        g.remove_marked()
        assert g.passband_names == ['g', 'r']
        assert_array_equal(g.pbids, array([0, 0, 1]))
        assert g.lcslices[-1] == slice(200, 300)

    def test_removal_does_not_touch_another_group_holding_the_same_light_curves(self):
        g = make_group()
        sub = g.select(instrument='MuSCAT2')
        g.mark_for_removal(1)
        g.remove_marked()
        assert g.size == 3 and sub.size == 3 and sub.n_marked == 1

    def test_the_group_still_feeds_baselpf_after_a_removal(self):
        lcs = make_group()
        lcs.mark_for_removal(0)
        lcs.remove_marked()
        lpf = BaseLPF('cut', passbands=lcs.passband_names, times=lcs.times, fluxes=lcs.fluxes,
                      pbids=lcs.pbids, covariates=lcs.covariates, wnids=lcs.wnids,
                      tm=RoadRunnerModel('quadratic'))
        assert lpf.nlc == 3 and lpf.timea.size == 300


class TestPlotting:
    def test_returns_a_figure_with_one_axis_per_light_curve(self):
        fig = make_group().plot()
        assert len(fig.axes) == 4
        close(fig)

    def test_grid_geometry(self):
        fig = make_group().plot(ncols=2)
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
        close(fig)

    def test_ncols_is_clipped_to_the_light_curve_count(self):
        fig = make_group().plot(ncols=10)
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (1, 4)
        close(fig)

    def test_leftover_axes_are_removed(self):
        fig = make_group()[:3].plot(ncols=2)
        assert fig.axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 2)
        assert len(fig.axes) == 3
        close(fig)

    def test_y_limits_are_shared(self):
        fig = make_group().plot()
        assert fig.axes[0].get_ylim() == fig.axes[-1].get_ylim()
        close(fig)

    def test_ylim_is_honoured(self):
        fig = make_group().plot(ylim=(0.99, 1.01))
        assert all(ax.get_ylim() == (0.99, 1.01) for ax in fig.axes)
        close(fig)

    def test_passband_filtering(self):
        g = make_group()
        fig = g.plot(passbands='g')
        assert len(fig.axes) == 2
        close(fig)
        fig = g.plot(passbands=['g', 'r'])
        assert len(fig.axes) == 3
        close(fig)

    def test_instrument_and_sector_filtering(self):
        g = make_group()
        fig = g.plot(instruments='MuSCAT2')
        assert len(fig.axes) == 3
        close(fig)
        fig = g.plot(sectors=1)
        assert len(fig.axes) == 1
        close(fig)

    def test_pid_filtering(self):
        g = LCDataGroup([make_lc(pids=(0, 1), seed=0), make_lc(pids=2, t0=1.0, seed=1)])
        fig = g.plot(pids=[0, 2])
        assert len(fig.axes) == 2
        close(fig)

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match='No light curves to plot matching'):
            make_group().plot(passbands='nonexistent')

    def test_empty_group_raises(self):
        with pytest.raises(ValueError, match='the group is empty'):
            LCDataGroup().plot()

    def test_invalid_ncols_raises(self):
        with pytest.raises(ValueError, match="'ncols' must be at least one"):
            make_group().plot(ncols=0)

    def test_annotation_shows_the_instrument_and_the_passband(self):
        fig = make_group().plot(show_index=False)
        assert [t.get_text() for t in fig.axes[0].texts] == ['TESS\nTESS']
        assert [t.get_text() for t in fig.axes[1].texts] == ['MuSCAT2\ng']
        close(fig)

    def test_annotation_can_be_switched_off(self):
        fig = make_group().plot(annotate=False, show_index=False)
        assert all(len(ax.texts) == 0 for ax in fig.axes)
        close(fig)

    def test_annotation_omits_an_empty_instrument(self):
        fig = LCDataGroup([make_lc(instrument='')]).plot(show_index=False)
        assert [t.get_text() for t in fig.axes[0].texts] == ['TESS']
        close(fig)

    def test_annotation_joins_multiple_passbands(self):
        fig = LCDataGroup([make_lc(passband=('g', 'r'), instrument='')]).plot(show_index=False)
        assert [t.get_text() for t in fig.axes[0].texts] == ['g+r']
        close(fig)

    def test_times_are_offset_per_panel_by_default(self):
        g = make_group()
        fig = g.plot()
        for ax, lc in zip(fig.axes, g):
            assert_allclose(ax.lines[0].get_xdata(), lc.time - floor(lc.time.min()))
        assert fig.axes[0].get_xlabel() == 'Time - 0 [BJD]'
        assert fig.axes[2].get_xlabel() == 'Time - 2 [BJD]'
        close(fig)

    def test_zero_offset_plots_the_times_as_they_are(self):
        g = make_group()
        fig = g.plot(xoffset=0.0)
        assert_allclose(fig.axes[0].lines[0].get_xdata(), g[0].time)
        assert fig.axes[0].get_xlabel() == 'Time [BJD]'
        close(fig)

    def test_common_offset_labels_the_bottom_row_only(self):
        g = make_group()
        fig = g.plot(ncols=2, xoffset=1.0)
        assert_allclose(fig.axes[0].lines[0].get_xdata(), g[0].time - 1.0)
        assert fig.axes[0].get_xlabel() == ''
        assert fig.axes[2].get_xlabel() == 'Time - 1 [BJD]'
        close(fig)

    def test_errorbars(self):
        fig = make_group().plot(errorbars=True)
        assert len(fig.axes) == 4
        assert fig.axes[0].containers
        close(fig)

    def test_errorbars_with_explicit_errors(self):
        fig = LCDataGroup([make_lc(error=full(NPT, 1e-3))]).plot(errorbars=True)
        assert fig.axes[0].containers
        close(fig)

    def test_kwargs_reach_the_plot_call(self):
        fig = make_group().plot(marker='o', color='k')
        assert fig.axes[0].lines[0].get_marker() == 'o'
        close(fig)

    def test_the_light_curve_index_is_shown(self):
        fig = make_group().plot()
        assert '0' in [t.get_text() for t in fig.axes[0].texts]
        assert '3' in [t.get_text() for t in fig.axes[3].texts]
        close(fig)

    def test_the_shown_index_is_the_group_index_not_the_panel_index(self):
        fig = make_group().plot(passbands='r')     # Only light curve 3 survives the filter.
        assert [t.get_text() for t in fig.axes[0].texts if t.get_text().isdigit()] == ['3']
        close(fig)

    def test_the_index_survives_annotate_false(self):
        fig = make_group().plot(annotate=False)
        assert [t.get_text() for t in fig.axes[2].texts] == ['2']
        close(fig)

    def test_xticks_can_be_switched_off(self):
        fig = make_group().plot(show_xticks=False)
        assert all(len(ax.get_xticks()) == 0 and ax.get_xlabel() == '' for ax in fig.axes)
        close(fig)

    def test_xticks_can_be_switched_off_with_a_common_offset(self):
        fig = make_group().plot(ncols=2, xoffset=0.0, show_xticks=False)
        assert all(len(ax.get_xticks()) == 0 and ax.get_xlabel() == '' for ax in fig.axes)
        close(fig)

    def test_marked_light_curves_get_a_gray_background(self):
        g = make_group()
        g.mark_for_removal(1)
        fig = g.plot()
        assert_allclose(fig.axes[1].get_facecolor(), (0.9, 0.9, 0.9, 1.0))
        assert fig.axes[0].get_facecolor() != fig.axes[1].get_facecolor()
        close(fig)

    def test_show_median_draws_a_band_per_sigma_level(self):
        g = make_group()
        fig = g.plot(show_median=True, nsigma=(1, 3, 5))
        assert all(len(ax.collections) == 3 for ax in fig.axes)
        close(fig)

    def test_show_median_accepts_a_scalar_nsigma(self):
        fig = make_group().plot(show_median=True)
        assert all(len(ax.collections) == 1 for ax in fig.axes)
        close(fig)

    def test_nothing_is_drawn_without_show_median(self):
        fig = make_group().plot()
        assert all(len(ax.collections) == 0 and len(ax.lines) == 1 for ax in fig.axes)
        close(fig)

    def test_the_median_is_drawn_after_the_data_at_the_panel_offset(self):
        g = make_group()
        fig = g.plot(show_median=True)
        for ax, lc in zip(fig.axes, g):
            assert_allclose(ax.lines[0].get_xdata(), lc.time - floor(lc.time.min()))   # Data first.
            assert_allclose(ax.lines[-1].get_ydata(), lc.running_median(15))
            assert_allclose(ax.lines[-1].get_xdata(), lc.time - floor(lc.time.min()))
        close(fig)

    def test_show_linear_model_draws_the_covariate_fit(self):
        g = make_group()
        fig = g.plot(show_linear_model=True)
        for ax, lc in zip(fig.axes, g):
            assert len(ax.lines) == 2                                      # Data plus the model.
            assert_allclose(ax.lines[0].get_ydata(), lc.flux)              # The data stays first.
            assert_allclose(ax.lines[-1].get_ydata(), lc.linear_model())
        close(fig)

    def test_show_linear_model_skips_light_curves_without_covariates(self):
        g = LCDataGroup([make_lc(ncov=0)])
        fig = g.plot(show_linear_model=True)
        assert len(fig.axes[0].lines) == 1
        close(fig)

    def test_the_overlays_are_drawn_above_the_data(self):
        fig = make_group().plot(show_median=True, show_linear_model=True)
        for ax in fig.axes:
            data, overlays = ax.lines[0], ax.lines[1:]
            assert len(overlays) == 2
            assert all(o.get_zorder() > data.get_zorder() for o in overlays)
        close(fig)

    def test_the_overlay_line_properties_can_be_customised(self):
        fig = make_group().plot(show_median=True, show_linear_model=True,
                                median_kwargs=dict(c='C0', lw=2, alpha=0.6),
                                linear_model_kwargs=dict(c='r', lw=3, alpha=0.8, zorder=20))
        med, lm = fig.axes[0].lines[1], fig.axes[0].lines[2]
        assert (med.get_color(), med.get_linewidth(), med.get_alpha()) == ('C0', 2.0, 0.6)
        assert (lm.get_color(), lm.get_linewidth(), lm.get_alpha()) == ('r', 3.0, 0.8)
        assert lm.get_zorder() == 20
        close(fig)

    def test_the_bands_follow_the_median_colour(self):
        fig = make_group().plot(show_median=True, median_kwargs=dict(c='C0'))
        assert_allclose(fig.axes[0].collections[0].get_facecolor()[0],
                        to_rgba('C0', 0.15))
        close(fig)

    def test_the_default_overlay_styles_are_not_shared_between_calls(self):
        """The defaults are merged into a new dict, so an override cannot leak into the next call."""
        g = make_group()
        close(g.plot(show_median=True, median_kwargs=dict(c='r')))
        fig = g.plot(show_median=True)
        assert fig.axes[0].lines[1].get_color() == 'k'
        close(fig)

    def test_axis_labels(self):
        fig = make_group().plot(ncols=2)
        assert fig.axes[0].get_ylabel() == 'Normalised flux'
        assert fig.axes[1].get_ylabel() == ''
        close(fig)
