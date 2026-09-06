import numpy as np
from src.planning.trajectory import linear_joint_trajectory, quintic_joint_trajectory


def test_linear_trajectory_endpoints():
    q_start = [0, 0, 0, 0, 0, 0]
    q_target = [1, -1, 0.5, 0, 0.2, -0.3]
    traj = linear_joint_trajectory(q_start, q_target, steps=20)
    assert traj.shape == (20, 6)
    assert np.allclose(traj[0], q_start)
    assert np.allclose(traj[-1], q_target)


def test_linear_trajectory_is_monotonic_per_joint():
    traj = linear_joint_trajectory([0, 0], [2, -2], steps=10)
    diffs = np.diff(traj, axis=0)
    assert np.all(diffs[:, 0] >= 0)
    assert np.all(diffs[:, 1] <= 0)


def test_quintic_trajectory_endpoints():
    q_start = [0, 0, 0, 0, 0, 0]
    q_target = [1, 1, 1, 1, 1, 1]
    traj = quintic_joint_trajectory(q_start, q_target, steps=30)
    assert np.allclose(traj[0], q_start, atol=1e-9)
    assert np.allclose(traj[-1], q_target, atol=1e-9)


def test_quintic_starts_and_ends_slower_than_middle():
    traj = quintic_joint_trajectory([0], [1], steps=100)
    diffs = np.abs(np.diff(traj[:, 0]))
    assert diffs[0] < diffs[len(diffs) // 2]
    assert diffs[-1] < diffs[len(diffs) // 2]
