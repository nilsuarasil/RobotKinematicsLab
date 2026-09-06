"""
Inverse Kinematics: hedef bir end-effector konumu verilince, oraya
ulaştıran eklem açılarını (q1..q6) sayısal olarak bulur.

Yöntem: Damped Least Squares (Levenberg-Marquardt'a benzer), Jacobian
tabanlı iteratif çözüm:
    dq = J^T (J J^T + lambda^2 I)^-1 dx

Saf pseudo-inverse (dq = J^+ dx) yerine damping eklenmesinin sebebi:
singularity'lere yaklaşıldığında J^+ patlayabilir (çok büyük adımlar
üretir); damping bunu sayısal olarak stabilize eder.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.kinematics.forward import end_effector_position
from src.kinematics.jacobian import compute_jacobian
from src.robot.joint_limits import JOINT_LIMITS, check_joint_limits


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    final_error: float
    joint_limit_violations: List = field(default_factory=list)


def inverse_kinematics(
    target_position,
    q_init=None,
    max_iters: int = 500,
    tol: float = 1e-4,
    damping: float = 0.05,
    step_scale: float = 1.0,
    joint_limits=JOINT_LIMITS,
    n_joints: int = 6,
) -> IKResult:
    """
    Sadece konum hedefi (x, y, z) için IK çözer -- Jacobian'ın ilk 3 satırı
    (doğrusal hız kısmı) kullanılır. Yönelim (orientation) V1 kapsamı dışında.
    """
    target_position = np.asarray(target_position, dtype=float)
    q = np.array(q_init, dtype=float) if q_init is not None else np.zeros(n_joints)

    converged = False
    error_norm = float("inf")
    iteration = 0

    for iteration in range(max_iters):
        current_position = end_effector_position(q)
        error = target_position - current_position
        error_norm = float(np.linalg.norm(error))

        if error_norm < tol:
            converged = True
            break

        J_full = compute_jacobian(q)
        J = J_full[:3, :]  # sadece pozisyon satırları

        JJt = J @ J.T
        damped = JJt + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(damped, error)
        q = q + step_scale * dq

        # açıları [-pi, pi] aralığına sar (okunabilirlik için)
        q = (q + np.pi) % (2 * np.pi) - np.pi

    violations = check_joint_limits(q, joint_limits)

    return IKResult(
        q=q,
        converged=converged,
        iterations=iteration + 1,
        final_error=error_norm,
        joint_limit_violations=violations,
    )


def inverse_kinematics_generic(
    fk_func,
    n_joints: int,
    target_position,
    q_init=None,
    max_iters: int = 500,
    tol: float = 1e-4,
    damping: float = 0.05,
    step_scale: float = 1.0,
    joint_limits=JOINT_LIMITS,
    eps: float = 1e-6,
) -> IKResult:
    """
    DH tablosuna bağlı OLMAYAN, genel bir sayısal IK çözücü. `fk_func(q)`
    herhangi bir forward-kinematics fonksiyonu olabilir -- Jacobian'ı
    analitik formülle değil, merkezi fark (central difference) ile sayısal
    olarak tahmin eder. Bu, örneğin CoppeliaSim sahnesinden kalibre edilmiş
    (empirik) bir kinematik zincirle çalışmak için kullanılır -- bkz.
    robot_model.CalibratedUR5Robot -- çünkü o zincirin analitik bir Jacobian
    formülü yoktur (geometri koddan değil, sahneden okunur).

    Matematiksel yöntem `inverse_kinematics` ile birebir aynı (damped least
    squares); tek fark Jacobian'ın nasıl elde edildiği.
    """
    target_position = np.asarray(target_position, dtype=float)
    q = np.array(q_init, dtype=float) if q_init is not None else np.zeros(n_joints)

    converged = False
    error_norm = float("inf")
    iteration = 0

    for iteration in range(max_iters):
        current_position = np.asarray(fk_func(q), dtype=float)
        error = target_position - current_position
        error_norm = float(np.linalg.norm(error))

        if error_norm < tol:
            converged = True
            break

        J = np.zeros((3, n_joints))
        for i in range(n_joints):
            dq = np.zeros(n_joints)
            dq[i] = eps
            p_plus = np.asarray(fk_func(q + dq), dtype=float)
            p_minus = np.asarray(fk_func(q - dq), dtype=float)
            J[:, i] = (p_plus - p_minus) / (2 * eps)

        JJt = J @ J.T
        damped = JJt + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(damped, error)
        q = q + step_scale * dq
        q = (q + np.pi) % (2 * np.pi) - np.pi

    violations = check_joint_limits(q, joint_limits)

    return IKResult(
        q=q,
        converged=converged,
        iterations=iteration + 1,
        final_error=error_norm,
        joint_limit_violations=violations,
    )
