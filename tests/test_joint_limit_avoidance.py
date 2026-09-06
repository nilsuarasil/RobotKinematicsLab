import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.control.joint_limit_avoidance import (
    joint_limit_avoidance_gradient, null_space_projection,
    inverse_kinematics_with_joint_limit_avoidance,
)
from src.robot.joint_limits import JOINT_LIMITS


def test_gradient_zero_at_limit_center():
    q_mid = np.array([0.5 * (lo + hi) for lo, hi in JOINT_LIMITS])
    grad = joint_limit_avoidance_gradient(q_mid, JOINT_LIMITS)
    assert np.allclose(grad, 0.0, atol=1e-10)


def test_gradient_points_away_from_upper_limit():
    lo, hi = JOINT_LIMITS[0]
    q = np.array([hi - 0.01] + [0.0] * 5)  # J1 üst limite çok yakın
    grad = joint_limit_avoidance_gradient(q, JOINT_LIMITS)
    assert grad[0] > 0  # limite yaklaştıkça gradyan büyür (pozitif -- üst limit yönünde)


def test_null_space_projection_preserves_primary_task():
    """(I - J^+ J) z, J'nin null-space'inde olmalı -- yani J @ N @ z ~ 0,
    dolayısıyla J @ dq_total ~ J @ dq_primary (birincil görev bozulmaz)."""
    rng = np.random.default_rng(5)
    J = rng.uniform(-1, 1, size=(3, 6))
    dq_primary = rng.uniform(-1, 1, size=6)
    z = rng.uniform(-1, 1, size=6)

    dq_total = null_space_projection(J, dq_primary, z)
    assert np.allclose(J @ dq_total, J @ dq_primary, atol=1e-8)


def test_joint_limit_avoidance_ik_converges():
    q_true = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    result = inverse_kinematics_with_joint_limit_avoidance(
        target_position=target, q_init=np.zeros(6),
    )
    assert result.converged
    assert result.position_error < 1e-3


def test_joint_limit_avoidance_keeps_joints_closer_to_center():
    """Null-space avoidance AKTIF iken, eklemlerin limit-merkezine olan
    ortalama uzaklığı, avoidance KAPALI (gain=0) haline göre >= olmamalı
    (yani avoidance eklemleri merkeze çeker ya da en azından kötüleştirmez)."""
    q_true = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    result_with_avoidance = inverse_kinematics_with_joint_limit_avoidance(
        target_position=target, q_init=np.zeros(6), null_space_gain=2.0,
    )
    result_without = inverse_kinematics_with_joint_limit_avoidance(
        target_position=target, q_init=np.zeros(6), null_space_gain=0.0,
    )
    grad_with = joint_limit_avoidance_gradient(result_with_avoidance.q)
    grad_without = joint_limit_avoidance_gradient(result_without.q)
    assert np.linalg.norm(grad_with) <= np.linalg.norm(grad_without) + 1e-6
