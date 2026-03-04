"""Tests for the TransitModel base class.

Tests the constructor defaults, set_data input validation and state management,
and the abstract method contracts using a minimal concrete subclass.
"""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from pytransit.models.transitmodel import TransitModel


class ConcreteTransitModel(TransitModel):
    """Minimal concrete subclass for testing base class logic."""
    def _init_model(self):
        pass


class TestConstructor:
    """Test TransitModel.__init__ defaults and parameter passing."""

    def test_default_parameters(self):
        tm = ConcreteTransitModel()
        assert tm.backend == "numba"
        assert tm.return_grad is False
        assert tm.parallel is False
        assert tm.n_threads is None

    def test_custom_parameters(self):
        tm = ConcreteTransitModel(backend="jax", return_grad=True, parallel=True, n_threads=4)
        assert tm.backend == "jax"
        assert tm.return_grad is True
        assert tm.parallel is True
        assert tm.n_threads == 4

    def test_initial_state(self):
        tm = ConcreteTransitModel()
        assert tm.times is None
        assert tm.lcids is None
        assert tm.pbids is None
        assert tm.epids is None
        assert tm.nsamples is None
        assert tm.exptimes is None
        assert tm.nlc == 0
        assert tm.npt == 0
        assert tm.npb == 0
        assert tm.ntc == 0
        assert tm.nor == 0
        assert tm.simple is True
        assert tm._time_id is None

    def test_abstract_init_model(self):
        with pytest.raises(NotImplementedError):
            TransitModel()


class TestSetDataHappyPaths:
    """Test set_data with valid inputs."""

    def test_times_only(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 100)
        tm.set_data(times)
        assert tm.npt == 100
        assert tm.nlc == 1
        assert tm.npb == 1
        assert tm.ntc == 1
        assert tm.simple is True
        assert_allclose(tm.lcids, np.zeros(100, 'int'))
        assert_allclose(tm.pbids, np.zeros(1, 'int'))
        assert_allclose(tm.epids, np.zeros(1, 'int'))
        assert_allclose(tm.nsamples, np.ones(1, 'int'))
        assert_allclose(tm.exptimes, np.zeros(1, 'int'))

    def test_times_converted_to_float64(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 50, dtype=np.float32)
        tm.set_data(times)
        assert tm.times.dtype == np.float64

    def test_accepts_python_lists(self):
        tm = ConcreteTransitModel()
        tm.set_data([1.0, 2.0, 3.0])
        assert isinstance(tm.times, np.ndarray)
        assert tm.npt == 3

    def test_multi_lightcurve(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 60)
        lcids = np.array([0]*20 + [1]*20 + [2]*20)
        tm.set_data(times, lcids=lcids)
        assert tm.nlc == 3
        assert tm.npt == 60

    def test_multi_passband(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        pbids = np.array([0, 1])
        tm.set_data(times, lcids=lcids, pbids=pbids)
        assert tm.npb == 2
        assert tm.simple is False

    def test_multi_epoch(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        epids = np.array([0, 1])
        tm.set_data(times, lcids=lcids, epids=epids)
        assert tm.ntc == 2
        assert tm.simple is False

    def test_simple_single_passband_single_epoch(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        pbids = np.array([0, 0])
        epids = np.array([0, 0])
        tm.set_data(times, lcids=lcids, pbids=pbids, epids=epids)
        assert tm.simple is True

    def test_supersampling(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        tm.set_data(times, lcids=lcids, nsamples=[10, 5], exptimes=[0.02, 0.01])
        assert_allclose(tm.nsamples, [10, 5])
        assert_allclose(tm.exptimes, [0.02, 0.01])

    def test_include_orbit_variations(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        epids = np.array([0, 1])
        tm.set_data(times, lcids=lcids, epids=epids, include_orbit_variations=True)
        assert tm.nor == tm.ntc
        assert tm.nor == 2

    def test_orbit_variations_default_off(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        epids = np.array([0, 1])
        tm.set_data(times, lcids=lcids, epids=epids)
        assert tm.nor == 1

    def test_id_based_caching(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 50)
        tm.set_data(times)
        first_time_id = tm._time_id

        # Calling again with the same object should early-return
        tm.set_data(times)
        assert tm._time_id == first_time_id

    def test_id_caching_skipped_when_other_args_given(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 50)
        tm.set_data(times)

        # Same times object but with explicit lcids should NOT early-return
        lcids = np.zeros(50, 'int')
        tm.set_data(times, lcids=lcids)
        # Should still work (no error), nlc should be set
        assert tm.nlc == 1


class TestSetDataValidation:
    """Test set_data input validation errors."""

    def test_lcids_non_integer(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 50)
        with pytest.raises(ValueError, match="integers"):
            tm.set_data(times, lcids=np.zeros(50, 'float'))

    def test_lcids_size_mismatch(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 50)
        with pytest.raises(ValueError, match="number of datapoints"):
            tm.set_data(times, lcids=np.zeros(30, 'int'))

    def test_pbids_non_integer(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 50)
        with pytest.raises(ValueError, match="integers"):
            tm.set_data(times, pbids=np.array([0.0]))

    def test_pbids_size_mismatch(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        with pytest.raises(ValueError, match="number of light curves"):
            tm.set_data(times, lcids=lcids, pbids=np.array([0, 1, 2]))

    def test_pbids_non_contiguous(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        with pytest.raises(ValueError, match="integers between 0"):
            tm.set_data(times, lcids=lcids, pbids=np.array([0, 2]))

    def test_epids_size_mismatch(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        with pytest.raises(ValueError, match="number of light curves"):
            tm.set_data(times, lcids=lcids, epids=np.array([0, 1, 2]))

    def test_epids_non_contiguous(self):
        tm = ConcreteTransitModel()
        times = np.linspace(-0.1, 0.1, 40)
        lcids = np.array([0]*20 + [1]*20)
        with pytest.raises(ValueError, match="integers between 0"):
            tm.set_data(times, lcids=lcids, epids=np.array([0, 2]))


class TestEvaluate:
    """Test that evaluate raises NotImplementedError."""

    def test_evaluate_not_implemented(self):
        tm = ConcreteTransitModel()
        tm.set_data(np.linspace(-0.1, 0.1, 50))
        with pytest.raises(NotImplementedError):
            tm.evaluate(k=0.1, t0=0.0, p=1.0, a=5.0, i=1.5)
