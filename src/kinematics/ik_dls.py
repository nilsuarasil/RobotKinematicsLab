"""
IK Solver #2: Damped Least Squares (DLS) -- Levenberg-Marquardt'a benzer.

    dq = J^T (J J^T + lambda^2 I)^-1 e
    q_new = q + alpha * dq

Pseudoinverse'ün singularity yakınındaki patlama sorununu `lambda` (damping)
terimiyle sayısal olarak stabilize eder: J J^T tekilleşse (singular olsa)
bile (J J^T + lambda^2 I) hep tersinir kalır. Bedeli: lambda arttıkça
yakınsama yavaşlar (daha fazla iterasyon) ama adımlar küçük ve güvenli kalır
-- singularity yakınında ani/sert eklem hareketlerini engellediği için özellikle
hassas (örn. cerrahi) uygulamalarda pseudoinverse'e tercih edilir.
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


def inverse_kinematics_dls(
    target_position,
    target_orientation=None,
    q_init=None,
    dh_params=UR5_DH_PARAMS,
    max_iters: int = 500,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-3,
    damping: float = 0.05,
    alpha: float = 1.0,
    joint_limits=JOINT_LIMITS,
    n_joints: int = 6,
) -> PoseIKResult:
    target_R = resolve_target_rotation(target_orientation)
    use_orientation = target_R is not None
    m = 6 if use_orientation else 3

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
        JJt = J @ J.T
        damped = JJt + (damping ** 2) * np.eye(m)
        dq = J.T @ np.linalg.solve(damped, error)
        q = wrap_to_pi(q + alpha * dq)
    elapsed = time.perf_counter() - t_start

    violations = check_joint_limits(q, joint_limits)
    return PoseIKResult(
        q=q, converged=converged, iterations=iteration + 1,
        position_error=pos_err_norm, orientation_error=ori_err_norm,
        joint_limit_violations=violations,
        solver_name="Damped Least Squares", elapsed_time=elapsed,
    )
