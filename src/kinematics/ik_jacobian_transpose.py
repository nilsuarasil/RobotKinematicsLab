"""
IK Solver #3: Jacobian Transpose.

    dq = alpha * J^T e

Pseudoinverse/DLS gibi bir matris tersi (inverse/solve) HİÇ gerektirmez --
tek maliyeti bir matris çarpımıdır, bu yüzden çok hızlı ve singularity'de
asla patlamaz (J^T her zaman tanımlıdır). Bedeli: yakınsama pseudoinverse/DLS'e
göre çok daha yavaştır ve doğru `alpha` seçimi kritik.

Sabit bir alpha yerine, her iterasyonda "optimal" adım büyüklüğünü analitik
olarak hesaplıyoruz (Wolovich & Elliott formülü):

    alpha* = (e^T J J^T e) / (e^T J J^T J J^T e)

Bu, dq = alpha J^T e yönünde hatayı en hızlı azaltan adım büyüklüğüdür ve elle
ayarlanan sabit bir alpha'ya göre çok daha güvenilir yakınsama sağlar.
"""
import time
import numpy as np

from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.kinematics.jacobian import compute_jacobian
from src.kinematics.ik_result import PoseIKResult
from src.kinematics._ik_common import (
    resolve_target_rotation, pose_error_vector, select_jacobian_rows, wrap_to_pi,
)
from src.robot.joint_limits import JOINT_LIMITS, check_joint_limits


def inverse_kinematics_jacobian_transpose(
    target_position,
    target_orientation=None,
    q_init=None,
    dh_params=UR5_DH_PARAMS,
    max_iters: int = 1000,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-3,
    joint_limits=JOINT_LIMITS,
    n_joints: int = 6,
) -> PoseIKResult:
    target_R = resolve_target_rotation(target_orientation)
    use_orientation = target_R is not None

    q = np.array(q_init, dtype=float) if q_init is not None else np.zeros(n_joints)
    converged = False
    pos_err_norm = float("inf")
    ori_err_norm = 0.0
    iteration = 0

    t_start = time.perf_counter()
    for iteration in range(max_iters):
        T_current = forward_kinematics(q, dh_params)
        error, pos_err_norm, ori_err_norm = pose_error_vector(T_current, target_position, target_R)

        if pos_err_norm < pos_tol and (not use_orientation or ori_err_norm < ori_tol):
            converged = True
            break

        J = select_jacobian_rows(compute_jacobian(q, dh_params), use_orientation)
        JJt_e = J @ (J.T @ error)
        denom = float(JJt_e @ JJt_e)
        # Wolovich & Elliott optimal adım büyüklüğü. Payda ~0 ise (JJt_e ~ 0,
        # yani hata Jacobian'ın erişemediği bir yönde) küçük sabit bir adıma düş.
        if denom < 1e-12:
            alpha = 0.1
        else:
            alpha = float(error @ JJt_e) / denom

        dq = alpha * (J.T @ error)
        q = wrap_to_pi(q + dq)
    elapsed = time.perf_counter() - t_start

    violations = check_joint_limits(q, joint_limits)
    return PoseIKResult(
        q=q, converged=converged, iterations=iteration + 1,
        position_error=pos_err_norm, orientation_error=ori_err_norm,
        joint_limit_violations=violations,
        solver_name="Jacobian Transpose", elapsed_time=elapsed,
    )
