"""Unit tests for TransitModel parameter normalization helpers.

Tests the helper methods added to TransitModel base class for parameter
normalization and validation.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from pytransit.models.transitmodel import TransitModel


@pytest.fixture
def model():
    """A TransitModel with minimal data set so validation passes."""
    m = TransitModel()
    m.set_data(np.linspace(0, 1, 10))
    return m


@pytest.fixture
def model_centered():
    """A TransitModel with data centered on zero."""
    m = TransitModel()
    m.set_data(np.linspace(-0.1, 0.1, 10))
    return m


class TestTransitModelHelpers:
    """Test parameter normalization helper methods."""

    def test_detect_parameter_mode_scalar_float(self, model):
        """Test mode detection with scalar float period."""
        mode = model._detect_parameter_mode(1.0)
        assert mode == 'scalar'

    def test_detect_parameter_mode_scalar_t0(self, model):
        """Test mode detection with scalar t0 as fallback."""
        p = np.array([1.0])  # Array but check t0
        mode = model._detect_parameter_mode(p, t0=0.0)
        assert mode == 'scalar'

    def test_detect_parameter_mode_vector(self, model):
        """Test mode detection with vector period."""
        p = np.array([1.0, 1.1, 1.2])
        mode = model._detect_parameter_mode(p)
        assert mode == 'vector'

    def test_normalize_orbital_params_scalar_none(self, model):
        """Test orbital param normalization with None defaults (scalar)."""
        p = 1.0
        e, w = model._normalize_orbital_params(p, None, None, for_vector=False)
        assert e == 0.0
        assert w == 0.0

    def test_normalize_orbital_params_scalar_values(self, model):
        """Test orbital param normalization with explicit values (scalar)."""
        p = 1.0
        e, w = model._normalize_orbital_params(p, 0.3, 1.2, for_vector=False)
        assert e == 0.3
        assert w == 1.2

    def test_normalize_orbital_params_vector_none(self, model):
        """Test orbital param normalization with None defaults (vector)."""
        p = np.array([1.0, 1.1, 1.2])
        e, w = model._normalize_orbital_params(p, None, None, for_vector=True)
        assert_array_equal(e, np.zeros(3))
        assert_array_equal(w, np.zeros(3))

    def test_normalize_orbital_params_vector_values(self, model):
        """Test orbital param normalization with explicit values (vector)."""
        p = np.array([1.0, 1.1, 1.2])
        e_in = np.array([0.1, 0.2, 0.3])
        w_in = np.array([0.5, 1.0, 1.5])
        e, w = model._normalize_orbital_params(p, e_in, w_in, for_vector=True)
        assert_array_equal(e, e_in)
        assert_array_equal(w, w_in)

    def test_normalize_k_scalar(self, model):
        """Test k normalization for scalar mode."""
        k = 0.1
        k_norm = model._normalize_k(k, for_vector=False)
        assert isinstance(k_norm, np.ndarray)
        assert_array_equal(k_norm, np.array(0.1))

    def test_normalize_k_vector_1d(self, model):
        """Test k normalization reshaping 1D to 2D for vector mode."""
        k = np.array([0.09, 0.10, 0.11])
        k_norm = model._normalize_k(k, for_vector=True)
        assert k_norm.shape == (3, 1)
        assert_array_equal(k_norm.ravel(), k)

    def test_normalize_k_vector_2d(self, model):
        """Test k normalization preserving 2D shape for vector mode."""
        k = np.array([[0.09, 0.10], [0.10, 0.11], [0.11, 0.12]])
        k_norm = model._normalize_k(k, for_vector=True)
        assert k_norm.shape == (3, 2)
        assert_array_equal(k_norm, k)

    def test_normalize_t0_scalar(self, model):
        """Test t0 normalization for scalar mode."""
        t0 = 0.5
        t0_norm = model._normalize_t0(t0, for_vector=False)
        assert t0_norm == 0.5

    def test_normalize_t0_vector_1d(self, model):
        """Test t0 normalization reshaping 1D to 2D for vector mode."""
        t0 = np.array([0.0, 0.1, 0.2])
        t0_norm = model._normalize_t0(t0, for_vector=True)
        assert t0_norm.shape == (3, 1)
        assert_array_equal(t0_norm.ravel(), t0)

    def test_normalize_t0_vector_2d(self, model):
        """Test t0 normalization preserving 2D shape for vector mode."""
        t0 = np.array([[0.0], [0.1], [0.2]])
        t0_norm = model._normalize_t0(t0, for_vector=True)
        assert t0_norm.shape == (3, 1)
        assert_array_equal(t0_norm, t0)

    def test_normalize_ldc_2d(self, model):
        """Test ldc normalization with 2D input."""
        ldc = np.array([[0.2, 0.1], [0.3, 0.1]])
        ldc_norm = model._normalize_ldc(ldc, npv=2)
        assert ldc_norm.ndim == 2
        assert_array_equal(ldc_norm, ldc)

    def test_normalize_ldc_3d(self, model):
        """Test ldc normalization flattening 3D to 2D."""
        ldc = np.array([[[0.2, 0.1]], [[0.3, 0.1]], [[0.25, 0.15]]])
        ldc_norm = model._normalize_ldc(ldc, npv=3)
        assert ldc_norm.shape == (3, 2)
        assert_allclose(ldc_norm, np.array([[0.2, 0.1], [0.3, 0.1], [0.25, 0.15]]))

    def test_normalize_ldc_1d_to_2d(self, model):
        """Test ldc normalization converting 1D to 2D."""
        ldc = np.array([0.2, 0.1])
        ldc_norm = model._normalize_ldc(ldc, npv=1)
        assert ldc_norm.ndim == 2
        assert_array_equal(ldc_norm[0], ldc)

    def test_validate_time_set_with_data(self, model):
        """Test validation passes when data is set."""
        # Should not raise
        model._validate_time_set()

    def test_validate_time_set_without_data(self):
        """Test validation raises when data not set."""
        model = TransitModel()  # Fresh model without set_data
        with pytest.raises(ValueError, match="Need to set the data"):
            model._validate_time_set()


class TestEvaluateDeprecationWarnings:
    """Test deprecation warnings for evaluate_ps and evaluate_pv."""

    def test_evaluate_ps_deprecation_warning(self, model_centered):
        """Test that evaluate_ps issues deprecation warning."""
        with pytest.warns(DeprecationWarning, match=r"evaluate_ps\(\) is deprecated"):
            try:
                # This will fail with NotImplementedError, but should warn first
                model_centered.evaluate_ps(0.1, [0.2, 0.1], 0.0, 1.0, 3.0, 0.5 * np.pi)
            except NotImplementedError:
                pass  # Expected

    def test_evaluate_pv_deprecation_warning(self, model_centered):
        """Test that evaluate_pv issues deprecation warning."""
        with pytest.warns(DeprecationWarning, match=r"evaluate_pv\(\) may be deprecated"):
            try:
                # This will fail with NotImplementedError, but should warn first
                pvp = np.array([[0.1, 0.0, 1.0, 3.0, 0.5 * np.pi, 0.0, 0.0]])
                model_centered.evaluate_pv(pvp, [0.2, 0.1])
            except NotImplementedError:
                pass  # Expected


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError."""

    def test_call_backend_s_raises(self, model_centered):
        """Test that _call_backend_s raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="must implement _call_backend_s"):
            model_centered._call_backend_s(
                np.array(0.1), np.array([0.2, 0.1]), 0.0, 1.0, 3.0, 0.5 * np.pi, 0.0, 0.0
            )

    def test_call_backend_v_raises(self, model_centered):
        """Test that _call_backend_v raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="must implement _call_backend_v"):
            model_centered._call_backend_v(
                np.array([[0.1]]), np.array([[0.2, 0.1]]),
                np.array([[0.0]]), np.array([1.0]), np.array([3.0]),
                np.array([0.5 * np.pi]), np.array([0.0]), np.array([0.0])
            )
