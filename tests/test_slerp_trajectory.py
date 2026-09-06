import numpy as np
from src.kinematics.orientation import (
    rpy_to_rotation_matrix, rotation_matrix_to_quaternion, quaternion_normalize,
)
from src.planning.slerp import orientation_trajectory, cartesian_pose_trajectory


def test_orientation_trajectory_endpoints():
    q0 = quaternion_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    q1 = rotation_matrix_to_quaternion(rpy_to_rotation_matrix(0.3, 0.2, -0.5))
    traj = orientation_trajectory(q0, q1, steps=25)
    assert np.allclose(traj[0], q0, atol=1e-6)
    assert np.allclose(traj[-1], q1, atol=1e-6) or np.allclose(traj[-1], -q1, atol=1e-6)


def test_orientation_trajectory_stays_unit_norm():
    q0 = quaternion_normalize(np.array([1.0, 0.2, -0.1, 0.3]))
    q1 = quaternion_normalize(np.array([0.5, -0.5, 0.5, 0.5]))
    traj = orientation_trajectory(q0, q1, steps=30)
    norms = np.linalg.norm(traj, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_cartesian_pose_trajectory_shapes_and_endpoints():
    R0 = np.eye(3)
    R1 = rpy_to_rotation_matrix(0.1, 0.4, -0.2)
    p0, p1 = [0.0, 0.0, 0.0], [0.3, 0.1, 0.2]

    poses = cartesian_pose_trajectory(p0, p1, R0, R1, steps=20)
    assert len(poses) == 20
    assert np.allclose(poses[0][:3, 3], p0, atol=1e-9)
    assert np.allclose(poses[-1][:3, 3], p1, atol=1e-9)
    assert np.allclose(poses[0][:3, :3], R0, atol=1e-6)
    assert np.allclose(poses[-1][:3, :3], R1, atol=1e-5)
    for T in poses:
        # her ara transform da gecerli bir rotasyon matrisi icermeli
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
