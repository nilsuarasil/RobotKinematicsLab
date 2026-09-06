import numpy as np
from src.analysis.pose_error import position_error, position_error_mm


def test_zero_error_when_identical():
    assert position_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_known_distance():
    assert np.isclose(position_error([0, 0, 0], [3, 4, 0]), 5.0)


def test_mm_conversion():
    assert np.isclose(position_error_mm([0, 0, 0], [0.001, 0, 0]), 1.0)
