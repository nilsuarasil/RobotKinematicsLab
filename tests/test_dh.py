import numpy as np
from src.kinematics.dh import dh_transform, UR5_DH_PARAMS
from src.kinematics.rotations import is_rotation_matrix


def test_dh_transform_is_valid_rigid_transform():
    for a, alpha, d in UR5_DH_PARAMS:
        T = dh_transform(a, alpha, d, theta=0.5)
        assert is_rotation_matrix(T[:3, :3])
        assert np.allclose(T[3, :], [0, 0, 0, 1])


def test_dh_transform_zero_case():
    T = dh_transform(0, 0, 0, 0)
    assert np.allclose(T, np.eye(4))


def test_dh_transform_pure_translation_a():
    T = dh_transform(a=2.0, alpha=0, d=0, theta=0)
    assert np.allclose(T[:3, 3], [2.0, 0, 0])


def test_dh_transform_pure_translation_d():
    T = dh_transform(a=0, alpha=0, d=3.0, theta=0)
    assert np.allclose(T[:3, 3], [0, 0, 3.0])


def test_ur5_dh_table_has_six_joints():
    assert len(UR5_DH_PARAMS) == 6
