import numpy as np
from src.control.motion_scaling import scale_motion, MotionScaler, SCALE_PRESETS


def test_scale_motion_basic():
    result = scale_motion([10.0, -10.0, 5.0], 0.2)
    assert np.allclose(result, [2.0, -2.0, 1.0])


def test_presets_are_reciprocal_of_ratio():
    assert np.isclose(SCALE_PRESETS["5:1"], 1 / 5)
    assert np.isclose(SCALE_PRESETS["10:1"], 1 / 10)
    assert np.isclose(SCALE_PRESETS["1:1"], 1.0)


def test_motion_scaler_from_preset():
    scaler = MotionScaler.from_preset("5:1")
    result = scaler.apply(np.array([50.0, 0.0, 0.0]))
    assert np.allclose(result, [10.0, 0.0, 0.0])


def test_motion_scaler_set_scale_changes_behavior():
    scaler = MotionScaler(scale=1.0)
    assert np.allclose(scaler.apply([1.0]), [1.0])
    scaler.set_scale(0.5)
    assert np.allclose(scaler.apply([1.0]), [0.5])


def test_unknown_preset_raises():
    import pytest
    with pytest.raises(ValueError):
        MotionScaler.from_preset("3:7")
