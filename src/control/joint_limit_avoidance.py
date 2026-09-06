"""
Joint Limit Avoidance -- Null-space projection ile "ikincil hedef" (secondary
objective).

Normal IK (DLS/pseudoinverse) sadece BİRİNCİL hedefi (end-effector pose'u)
önemser -- eklemlerin limitlerine ne kadar yakın olduğunu umursamaz. 6-DOF'lu
(redundant olmayan) bir kolda bile, birden fazla eklem kombinasyonu aynı
pose'a ulaşabilir (bkz. multi_start_ik.py); bunlardan bazıları eklemleri
limitlerine (örn. J2 = 89.8°/90° limit) yapıştırabilir.

Null-space projection, birincil görevi (pose hatası) BOZMADAN, Jacobian'ın
null-space'inde ("boşa harcanan" serbestlik derecelerinde) ikincil bir
hedefi -- burada "eklemleri limit aralığının ORTASINA doğru it" -- uygular:

    dq = J^+ e  +  (I - J^+ J) z

Burada z, ikincil hedefin (H) gradyanına göre seçilir; H küçüldükçe eklemler
limit merkezine yaklaşır (Chiaverini'nin klasik formülasyonu):

    H(q) = (1 / 2n) * sum_i ( (q_i - q_mid_i) / (q_max_i - q_min_i) )^2
    z = -k * grad(H)
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


def joint_limit_avoidance_gradient(q, joint_limits=JOINT_LIMITS):
    """
    grad(H)_i = (1/n) * (q_i - q_mid_i) / (q_max_i - q_min_i)^2

    q, limit ortasındayken 0; bir limite yaklaştıkça büyür (işareti, hangi
    limite yaklaşıldığını gösterir).
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    grad = np.zeros(n)
    for i, (lo, hi) in enumerate(joint_limits):
        mid = 0.5 * (lo + hi)
        rng = hi - lo
        grad[i] = (q[i] - mid) / (rng ** 2) / n
    return grad


def null_space_projection(J, dq_primary, z):
    """dq_total = dq_primary + (I - J^+ J) z -- z, birincil görevi bozmadan
    "boşta kalan" serbestlik derecelerine projekte edilir."""
    n = J.shape[1]
    J_pinv = np.linalg.pinv(J)
    N = np.eye(n) - J_pinv @ J
    return dq_primary + N @ z


def inverse_kinematics_with_joint_limit_avoidance(
    target_position,
    target_orientation=None,
    q_init=None,
    dh_params=UR5_DH_PARAMS,
    max_iters: int = 500,
    pos_tol: float = 1e-4,
    ori_tol: float = 1e-3,
    damping: float = 0.05,
    alpha: float = 1.0,
    null_space_gain: float = 0.5,
    joint_limits=JOINT_LIMITS,
    n_joints: int = 6,
) -> PoseIKResult:
    """
    DLS'in birincil görevine (pose hatası), null-space'te joint-limit-avoidance
    ikincil hedefini ekleyen IK çözücü. Sonuç: hedefe DLS ile aynı doğrulukta
    ulaşır, ama eklemler mümkün olduğunca limit aralığının ortasına yakın kalır.
    """
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
        dq_primary = J.T @ np.linalg.solve(damped, error)

        z = -null_space_gain * joint_limit_avoidance_gradient(q, joint_limits)
        dq = null_space_projection(J, dq_primary, z)

        q = wrap_to_pi(q + alpha * dq)
    elapsed = time.perf_counter() - t_start

    violations = check_joint_limits(q, joint_limits)
    return PoseIKResult(
        q=q, converged=converged, iterations=iteration + 1,
        position_error=pos_err_norm, orientation_error=ori_err_norm,
        joint_limit_violations=violations,
        solver_name="DLS + Joint Limit Avoidance (Null-Space)", elapsed_time=elapsed,
    )
