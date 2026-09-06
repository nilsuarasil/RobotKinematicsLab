import numpy as np
import pytest
from src.kinematics.forward import forward_kinematics, forward_kinematics_all, end_effector_position
from src.kinematics.rotations import is_rotation_matrix


def test_fk_returns_valid_rigid_transform():
    q = [0.1, -0.5, 0.3, 0.2, -0.1, 0.4]
    T = forward_kinematics(q)
    assert T.shape == (4, 4)
    assert is_rotation_matrix(T[:3, :3])
    assert np.allclose(T[3, :], [0, 0, 0, 1])


def test_fk_all_returns_six_cumulative_transforms():
    q = [0, 0, 0, 0, 0, 0]
    transforms = forward_kinematics_all(q)
    assert len(transforms) == 6
    assert np.allclose(transforms[-1], forward_kinematics(q))


def test_fk_zero_config_is_finite_and_reasonable():
    pos = end_effector_position([0, 0, 0, 0, 0, 0])
    assert np.all(np.isfinite(pos))
    assert np.linalg.norm(pos) < 1.5  # UR5 link uzunlukları mertebesinde


def test_fk_changes_with_joint_angles():
    q1 = [0, 0, 0, 0, 0, 0]
    q2 = [0.5, 0, 0, 0, 0, 0]
    assert not np.allclose(end_effector_position(q1), end_effector_position(q2))


def test_fk_wrong_number_of_joints_raises():
    with pytest.raises(ValueError):
        forward_kinematics([0, 0, 0])
