import numpy as np
from src.planning.workspace import is_reachable, UR5_MAX_REACH


def test_target_within_reach():
    reachable, dist = is_reachable([0.4, 0.2, 0.3])
    assert reachable
    assert dist < UR5_MAX_REACH


def test_target_too_far():
    reachable, dist = is_reachable([4.0, 0.0, 0.0])
    assert not reachable
    assert dist > UR5_MAX_REACH


def test_target_too_close():
    reachable, _ = is_reachable([0.001, 0.0, 0.0])
    assert not reachable


def test_reachability_relative_to_custom_base():
    reachable, dist = is_reachable([1.0, 0.0, 0.0], base_position=[0.7, 0.0, 0.0])
    assert reachable
    assert np.isclose(dist, 0.3)
