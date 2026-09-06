import numpy as np
from src.planning.quintic import (
    quintic_time_scaling, quintic_time_scaling_derivative,
    quintic_time_scaling_second_derivative, quintic_scaling_profile,
)


def test_endpoints():
    assert np.isclose(quintic_time_scaling(0.0), 0.0)
    assert np.isclose(quintic_time_scaling(1.0), 1.0)


def test_velocity_zero_at_endpoints():
    assert np.isclose(quintic_time_scaling_derivative(0.0), 0.0, atol=1e-10)
    assert np.isclose(quintic_time_scaling_derivative(1.0), 0.0, atol=1e-10)


def test_acceleration_zero_at_endpoints():
    assert np.isclose(quintic_time_scaling_second_derivative(0.0), 0.0, atol=1e-10)
    assert np.isclose(quintic_time_scaling_second_derivative(1.0), 0.0, atol=1e-10)


def test_monotonic_increasing():
    profile = quintic_scaling_profile(100)
    assert np.all(np.diff(profile) >= 0)


def test_derivative_matches_finite_difference():
    t = np.linspace(0.01, 0.99, 50)
    eps = 1e-6
    analytic = quintic_time_scaling_derivative(t)
    numeric = (quintic_time_scaling(t + eps) - quintic_time_scaling(t - eps)) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-4)
