import numpy as np
from src.robot.joint_limits import JOINT_LIMITS, check_joint_limits, clamp_to_limits


def test_zero_config_within_limits():
    assert check_joint_limits([0, 0, 0, 0, 0, 0]) == []


def test_out_of_range_detected():
    q = [0, np.pi + 0.5, 0, 0, 0, 0]
    violations = check_joint_limits(q)
    assert len(violations) == 1
    assert violations[0][0] == 1


def test_clamp_to_limits():
    q = [10.0, -10.0, 0, 0, 0, 0]
    clamped = clamp_to_limits(q)
    assert clamped[0] == JOINT_LIMITS[0][1]
    assert clamped[1] == JOINT_LIMITS[1][0]
