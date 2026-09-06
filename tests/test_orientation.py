import numpy as np
from src.kinematics.orientation import (
    rpy_to_rotation_matrix, rotation_matrix_to_rpy,
    rotation_matrix_to_quaternion, quaternion_to_rotation_matrix,
    quaternion_multiply, quaternion_conjugate, quaternion_normalize,
    orientation_error, quaternion_error_norm, quaternion_slerp,
)
from src.kinematics.rotations import is_rotation_matrix


def test_rpy_round_trip():
    for roll, pitch, yaw in [(0.3, 0.2, 0.1), (-0.5, 0.4, -1.2), (0.0, 0.0, 0.0)]:
        R = rpy_to_rotation_matrix(roll, pitch, yaw)
        assert is_rotation_matrix(R)
        r2, p2, y2 = rotation_matrix_to_rpy(R)
        R2 = rpy_to_rotation_matrix(r2, p2, y2)
        assert np.allclose(R, R2, atol=1e-8)


def test_quaternion_rotation_matrix_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        roll, pitch, yaw = rng.uniform(-np.pi, np.pi, size=3)
        R = rpy_to_rotation_matrix(roll, pitch, yaw)
        q = rotation_matrix_to_quaternion(R)
        assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-10)
        R2 = quaternion_to_rotation_matrix(q)
        assert np.allclose(R, R2, atol=1e-8)


def test_quaternion_identity_multiply():
    q = np.array([0.7071, 0.0, 0.7071, 0.0])
    q = quaternion_normalize(q)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    result = quaternion_multiply(q, identity)
    assert np.allclose(result, q, atol=1e-8)


def test_quaternion_conjugate_inverts_rotation():
    q = quaternion_normalize(np.array([0.5, 0.5, 0.5, 0.5]))
    q_inv = quaternion_conjugate(q)
    result = quaternion_multiply(q, q_inv)
    assert np.allclose(result, [1, 0, 0, 0], atol=1e-8)


def test_orientation_error_zero_for_identical_rotations():
    R = rpy_to_rotation_matrix(0.4, -0.2, 1.1)
    err = orientation_error(R, R)
    assert np.allclose(err, np.zeros(3), atol=1e-10)


def test_orientation_error_matches_known_angle():
    from src.kinematics.rotations import rot_z
    R1 = np.eye(3)
    angle = 0.3
    R2 = rot_z(angle)
    err = orientation_error(R1, R2)
    assert np.isclose(np.linalg.norm(err), angle, atol=1e-8)
    # eksen Z olmalı
    assert np.allclose(err / np.linalg.norm(err), [0, 0, 1], atol=1e-6)


def test_quaternion_error_norm_nonnegative():
    R1 = rpy_to_rotation_matrix(0.1, 0.2, 0.3)
    R2 = rpy_to_rotation_matrix(-0.4, 0.5, 0.6)
    assert quaternion_error_norm(R1, R2) > 0


def test_slerp_endpoints():
    q0 = quaternion_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    q1 = quaternion_normalize(np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.allclose(quaternion_slerp(q0, q1, 0.0), q0, atol=1e-6)
    # t=1'de ya q1 ya da -q1 (aynı rotasyon) olmalı
    result = quaternion_slerp(q0, q1, 1.0)
    assert np.allclose(result, q1, atol=1e-6) or np.allclose(result, -q1, atol=1e-6)


def test_slerp_monotonic_angle():
    """SLERP boyunca q0'a olan açısal mesafe t ile monoton artmalı."""
    rng = np.random.default_rng(1)
    q0 = quaternion_normalize(rng.uniform(-1, 1, size=4))
    q1 = quaternion_normalize(rng.uniform(-1, 1, size=4))

    prev_dot = 1.0
    for t in np.linspace(0, 1, 10):
        q_t = quaternion_slerp(q0, q1, t)
        dot = abs(np.dot(q0, q_t))
        assert dot <= prev_dot + 1e-6
        prev_dot = dot
