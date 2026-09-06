import numpy as np
from src.kinematics.rotations import rot_z
from src.kinematics.transforms import (
    homogeneous_transform, transform_point, inverse_transform,
    position_from_transform, rotation_from_transform,
)


def test_identity_transform_leaves_point_unchanged():
    T = homogeneous_transform(np.eye(3), [0, 0, 0])
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(transform_point(T, p), p)


def test_pure_translation():
    T = homogeneous_transform(np.eye(3), [1, 2, 3])
    p = np.array([0.0, 0.0, 0.0])
    assert np.allclose(transform_point(T, p), [1, 2, 3])


def test_rotation_then_translation_order():
    T = homogeneous_transform(rot_z(np.pi / 2), [0, 0, 5])
    p = np.array([1.0, 0.0, 0.0])
    result = transform_point(T, p)
    assert np.allclose(result, [0, 1, 5], atol=1e-9)


def test_inverse_transform_round_trip():
    T = homogeneous_transform(rot_z(0.7), [1.5, -2.0, 3.3])
    T_inv = inverse_transform(T)
    assert np.allclose(T @ T_inv, np.eye(4), atol=1e-9)


def test_inverse_undoes_transform():
    T = homogeneous_transform(rot_z(1.1), [2.0, 0.5, -1.0])
    T_inv = inverse_transform(T)
    p = np.array([3.0, -1.0, 0.2])
    back = transform_point(T_inv, transform_point(T, p))
    assert np.allclose(back, p, atol=1e-9)


def test_extract_position_and_rotation():
    R = rot_z(0.3)
    t = np.array([1, 2, 3])
    T = homogeneous_transform(R, t)
    assert np.allclose(position_from_transform(T), t)
    assert np.allclose(rotation_from_transform(T), R)
