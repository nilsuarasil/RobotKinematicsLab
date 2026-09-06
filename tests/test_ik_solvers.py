"""
Üç V2 IK solver'ının (pseudoinverse, DLS, Jacobian transpose) hem
position-only hem de tam 6D pose (position+orientation) hedeflerde
yakınsadığını doğrular.
"""
import numpy as np
import pytest

from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.kinematics.orientation import rpy_to_rotation_matrix
from src.kinematics.ik_pseudoinverse import inverse_kinematics_pseudoinverse
from src.kinematics.ik_dls import inverse_kinematics_dls
from src.kinematics.ik_jacobian_transpose import inverse_kinematics_jacobian_transpose

SOLVERS = [
    inverse_kinematics_pseudoinverse,
    inverse_kinematics_dls,
    inverse_kinematics_jacobian_transpose,
]


@pytest.mark.parametrize("solver", SOLVERS)
def test_position_only_convergence(solver):
    q_true = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    result = solver(target_position=target, q_init=np.zeros(6))
    assert result.converged
    assert result.position_error < 1e-3

    achieved = forward_kinematics(result.q, UR5_DH_PARAMS)[:3, 3]
    assert np.linalg.norm(achieved - target) < 1e-3


@pytest.mark.parametrize("solver", SOLVERS)
def test_full_6d_pose_convergence(solver):
    q_true = np.array([0.2, -0.3, 0.4, 0.1, 0.5, -0.2])
    T_true = forward_kinematics(q_true, UR5_DH_PARAMS)
    target_position = T_true[:3, 3]
    target_R = T_true[:3, :3]

    result = solver(
        target_position=target_position, target_orientation=target_R,
        q_init=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        max_iters=2000, pos_tol=1e-3, ori_tol=5e-3,
    )
    assert result.converged, f"{result.solver_name} yakınsamadı"
    assert result.position_error < 2e-3
    assert result.orientation_error < 2e-2

    T_achieved = forward_kinematics(result.q, UR5_DH_PARAMS)
    assert np.linalg.norm(T_achieved[:3, 3] - target_position) < 2e-3


def test_target_orientation_accepts_rpy_and_quaternion():
    from src.kinematics.orientation import rotation_matrix_to_quaternion
    target_position = np.array([0.3, 0.2, 0.5])
    rpy = (0.4, 0.1, -0.2)
    R = rpy_to_rotation_matrix(*rpy)
    q = rotation_matrix_to_quaternion(R)

    result_rpy = inverse_kinematics_dls(
        target_position=target_position, target_orientation=np.array(rpy),
        q_init=np.zeros(6), max_iters=1000,
    )
    result_quat = inverse_kinematics_dls(
        target_position=target_position, target_orientation=q,
        q_init=np.zeros(6), max_iters=1000,
    )
    # İkisi de aynı hedefi (farklı temsille) çözmeli -- benzer hata seviyesine insin.
    assert result_rpy.position_error < 5e-3
    assert result_quat.position_error < 5e-3


def test_joint_limit_violations_reported():
    # Ulaşılamayacak kadar uç bir hedef vererek limit ihlali/az yakınsama olup
    # olmadığını (crash etmeden) kontrol ediyoruz -- asıl amaç hatasız çalışması.
    result = inverse_kinematics_dls(
        target_position=np.array([2.0, 2.0, 2.0]),  # workspace dışı
        q_init=np.zeros(6), max_iters=50,
    )
    assert isinstance(result.joint_limit_violations, list)
