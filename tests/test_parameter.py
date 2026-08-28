"""Tests for the model parameterization module."""

import pickle

import numpy as np
import pytest
from numpy import inf, isfinite, allclose

from pytransit.parameter import (Parameter, ParameterBlock, ParameterSet, normal_prior, uniform_prior,
                                 prior_kind, prior_limits, prior_moments, describe_prior)


def make_ps():
    ps = ParameterSet()
    ps.add_block('orbit', [Parameter('p', 'period', 'd', normal_prior(2.0, 1e-5), (0, inf)),
                           Parameter('rho', 'stellar density', 'g/cm^3', uniform_prior(1.0, 10.0), (0, inf)),
                           Parameter('b', 'impact parameter', '', uniform_prior(0.0, 1.0), (0, 1))])
    ps.add_block('area', [Parameter('k', 'radius ratio', '', uniform_prior(0.05, 0.15), (0, inf)),
                          Parameter('c', 'contamination', '', None, (0.0, 1.0))])
    ps.freeze()
    return ps


def test_prior_helpers():
    p = normal_prior(1.0, 0.5)
    assert prior_kind(p) == 'norm'
    assert prior_moments(p) == (1.0, 0.5)
    assert 'N(' in describe_prior(p)

    u = uniform_prior(0.2, 0.8)
    assert prior_kind(u) == 'uniform'
    assert allclose(prior_limits(u), (0.2, 0.8))
    assert allclose(u.mean(), 0.5)
    assert 'U(' in describe_prior(u)

    assert prior_kind(None) is None
    assert describe_prior(None) == '-'


def test_block_bookkeeping():
    ps = make_ps()
    assert [b.name for b in ps.blocks] == ['orbit', 'area']
    assert ps.blocks[0].start == 0 and ps.blocks[0].stop == 3
    assert ps.blocks[1].slice == slice(3, 5)
    assert ps.sl_orbit == slice(0, 3)
    assert ps.sl_area == slice(3, 5)
    assert [p.name for p in ps.bl_area] == ['k', 'c']
    assert [p.pid for p in ps] == list(range(5))


def test_duplicate_names():
    ps = make_ps()
    ps.thaw()
    with pytest.raises(ValueError):
        ps.add_block('orbit', [Parameter('x')])
    with pytest.raises(ValueError):
        ps.add_block('other', [Parameter('p')])


def test_log_prior():
    ps = make_ps()
    pv = np.array([2.0, 5.0, 0.5, 0.1, 0.5])
    lnp = ps.log_prior(pv)
    assert isfinite(lnp)

    # Out of bounds -> -inf
    bad = pv.copy()
    bad[2] = 1.5
    assert ps.log_prior(bad) == -inf

    # 2D input
    lnp2 = ps.log_prior(np.tile(pv, (3, 1)))
    assert lnp2.shape == (3,)
    assert allclose(lnp2, lnp)

    # The None prior contributes zero: changing c inside its bounds leaves lnp unchanged
    pv2 = pv.copy()
    pv2[4] = 0.9
    assert allclose(ps.log_prior(pv2), lnp)


def test_log_prior_gradient():
    ps = make_ps()
    pv = np.array([2.0 + 2e-6, 5.0, 0.5, 0.1, 0.5])
    g = ps.log_prior_gradient(pv)
    assert g.shape == (5,)

    h = 1e-8
    for i in range(5):
        pp, pm = pv.copy(), pv.copy()
        pp[i] += h
        pm[i] -= h
        fd = (ps.log_prior(pp) - ps.log_prior(pm)) / (2 * h)
        assert allclose(fd, g[i], atol=1e-4), f'parameter {ps[i].name}'

    ps[1].prior = pytest.importorskip('scipy.stats').gamma(2.0)
    with pytest.raises(NotImplementedError):
        ps.log_prior_gradient(pv)


def test_mean_pv():
    ps = make_ps()
    mp = ps.mean_pv
    assert allclose(mp, [2.0, 5.5, 0.5, 0.1, 0.5])

    ps2 = ParameterSet()
    ps2.add_parameter('x', prior=None, bounds=(-inf, inf))
    ps2.freeze()
    with pytest.raises(ValueError):
        ps2.mean_pv


def test_sample_from_prior():
    ps = make_ps()
    s = ps.sample_from_prior(100)
    assert s.shape == (100, 5)
    assert isfinite(ps.log_prior(s)).all()
    # The None-prior parameter samples uniformly within its bounds
    assert (s[:, 4] >= 0.0).all() and (s[:, 4] <= 1.0).all()


def test_set_prior():
    ps = make_ps()
    ps.set_prior('b', 'UP', 0.1, 0.4)
    assert allclose(prior_limits(ps[2].prior), (0.1, 0.4))
    ps.set_prior('b', 'normal', 0.2, 0.05)
    assert prior_moments(ps[2].prior) == (0.2, 0.05)
    ps.set_prior('b', normal_prior(0.3, 0.1))
    assert prior_moments(ps[2].prior) == (0.3, 0.1)
    with pytest.raises(ValueError):
        ps.set_prior('b', 'x', 0, 1)
    with pytest.raises(ValueError):
        ps.set_prior('b', 'u', 0.1)
    with pytest.raises(KeyError):
        ps.set_prior('nonexistent', 'u', 0, 1)


def test_set_bounds():
    ps = make_ps()
    ps.set_bounds('b', (0.0, 0.5))
    assert ps[2].bounds == (0.0, 0.5)
    assert ps.bounds[2, 1] == 0.5
    assert ps.frozen


def test_empty_set_freeze():
    ps = ParameterSet()
    ps.freeze()
    assert ps.bounds.shape == (0, 2)


def test_parameter_pickle_roundtrip():
    p = Parameter('k', 'radius ratio', '', uniform_prior(0.05, 0.15), (0, inf))
    q = pickle.loads(pickle.dumps(p))
    assert q.name == p.name
    assert allclose(prior_limits(q.prior), prior_limits(p.prior))
    assert q.bounds == p.bounds


def test_string_representations():
    ps = make_ps()
    for p in ps:
        assert isinstance(str(p), str)
    ps.check_pv(ps.mean_pv)


class TestVectorizedPriorParity:
    """log_prior/log_prior_gradient use cached vectorized descriptors; they must
    match the per-parameter scipy loop exactly, and the cache must refresh when
    a prior is replaced (via set_prior or by direct assignment)."""

    def _reference(self, ps, pv):
        pv = np.atleast_2d(pv)
        lnp = np.zeros(pv.shape[0])
        m = (pv > ps.bounds[:, 0]).all(1) & (pv < ps.bounds[:, 1]).all(1)
        for i, p in enumerate(ps):
            lnp = lnp + p.log_prior(pv[:, i])
        return np.squeeze(np.where(m, lnp, -np.inf))

    def _make(self):
        ps = ParameterSet()
        ps.add_block('a', [Parameter('x1', prior=normal_prior(1.0, 0.3)),
                           Parameter('x2', prior=uniform_prior(-2.0, 3.0), bounds=(-2, 3)),
                           Parameter('x3', prior=None, bounds=(0, 10)),
                           Parameter('x4', prior=uniform_prior(0.0, 1.0), bounds=(0, 1))])
        ps.freeze()
        return ps

    def test_log_prior_matches_scipy_loop(self):
        ps = self._make()
        rng = np.random.default_rng(1)
        pv = rng.normal([1, 0.5, 5, 0.5], [0.5, 2.0, 3.0, 0.6], (500, 4))
        new, ref = ps.log_prior(pv), self._reference(ps, pv)
        fin = np.isfinite(ref)
        assert np.array_equal(np.isfinite(new), fin)
        np.testing.assert_allclose(new[fin], ref[fin], rtol=1e-12)

    def test_prior_cache_refreshes(self):
        ps = self._make()
        x = np.array([1.0, 0.5, 5.0, 0.5])
        ps.log_prior(x)                       # populate the cache
        ps[0].prior = normal_prior(2.0, 0.1)  # direct assignment
        assert np.isclose(float(ps.log_prior(x)), float(self._reference(ps, x)))
        ps.set_prior('x2', 'N', 0.0, 1.0)
        assert np.isclose(float(ps.log_prior(x)), float(self._reference(ps, x)))

    def test_log_prior_gradient_normal_only(self):
        ps = self._make()
        x = np.array([1.1, 0.5, 5.0, 0.5])
        g = ps.log_prior_gradient(x)
        np.testing.assert_allclose(g, [-(1.1 - 1.0) / 0.3 ** 2, 0.0, 0.0, 0.0], atol=1e-12)
