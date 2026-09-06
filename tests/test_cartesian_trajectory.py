import numpy as np
from src.planning.cartesian_trajectory import cartesian_trajectory


def test_endpoints_match():
    p0, p1 = [0.1, 0.2, 0.3], [0.4, -0.1, 0.5]
    traj = cartesian_trajectory(p0, p1, steps=40)
    assert np.allclose(traj[0], p0, atol=1e-9)
    assert np.allclose(traj[-1], p1, atol=1e-9)


def test_is_a_straight_line():
    """Her ara nokta, p0->p1 doğrusu üzerinde olmalı (Cartesian uzayda düz çizgi)."""
    p0, p1 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, -1.0])
    traj = cartesian_trajectory(p0, p1, steps=30)
    direction = (p1 - p0) / np.linalg.norm(p1 - p0)
    for p in traj:
        v = p - p0
        proj = np.dot(v, direction) * direction
        perpendicular_dist = np.linalg.norm(v - proj)
        assert perpendicular_dist < 1e-9


def test_slower_at_ends_than_middle():
    traj = cartesian_trajectory([0, 0, 0], [1, 0, 0], steps=100)
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    assert diffs[0] < diffs[len(diffs) // 2]
    assert diffs[-1] < diffs[len(diffs) // 2]
