import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.jacobian import compute_jacobian
from src.control.resolved_rate import resolved_rate_joint_velocity, simulate_resolved_rate_motion


def test_resolved_rate_velocity_reproduces_cartesian_velocity():
    """q_dot = J^+ x_dot ise, J @ q_dot ~ x_dot olmali (x_dot, J'nin satir
    uzayinda/erisilebilir bir hiz oldugu surece)."""
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    J = compute_jacobian(q, UR5_DH_PARAMS)[:3, :]
    x_dot = np.array([0.01, -0.02, 0.005])

    q_dot = resolved_rate_joint_velocity(J, x_dot)
    assert np.allclose(J @ q_dot, x_dot, atol=1e-8)


def test_damped_resolved_rate_close_to_undamped_away_from_singularity():
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    J = compute_jacobian(q, UR5_DH_PARAMS)[:3, :]
    x_dot = np.array([0.01, 0.0, 0.0])

    q_dot_pinv = resolved_rate_joint_velocity(J, x_dot, damping=0.0)
    q_dot_dls = resolved_rate_joint_velocity(J, x_dot, damping=1e-3)
    assert np.allclose(q_dot_pinv, q_dot_dls, atol=1e-2)


def test_simulate_constant_velocity_moves_ee_in_expected_direction():
    q_start = np.zeros(6)
    velocity = np.array([0.02, 0.0, 0.0])  # +X yönünde 2 cm/s

    q_hist, ee_hist = simulate_resolved_rate_motion(
        q_start, velocity, duration=0.5, dt=0.01, damping=0.05,
    )
    displacement = ee_hist[-1] - ee_hist[0]
    # tam olarak 0.01 m degil (Jacobian yol boyunca degisir) ama X yonunde
    # baskin, pozitif bir hareket olmali ve makul bir mesafede kalmali.
    assert displacement[0] > 0.005
    assert np.linalg.norm(displacement) < 0.05
    assert q_hist.shape[1] == 6
