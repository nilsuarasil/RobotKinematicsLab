"""
Resolved-Rate Motion Control: V1/V2'nin diğer IK'ları "hedef POZİSYON verilince
hedef eklem AÇILARINI bul" derken, resolved-rate "hedef Cartesian HIZ verilince
gereken eklem HIZLARINI bul" der:

    x_dot = J q_dot   =>   q_dot = J^+ x_dot   (pseudoinverse ile "resolved")

Bu, teleoperation (mouse/gamepad ile sürükleme) ve sürekli hız komutuyla
kontrol (velocity control) için temel yapı taşıdır -- pozisyon IK'sının
aksine, "şu an hangi yöne, ne hızla hareket etmeliyim" sorusuna cevap verir.
"""
import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.kinematics.jacobian import compute_jacobian


def resolved_rate_joint_velocity(J, cartesian_velocity, damping: float = 0.0):
    """
    q_dot = J^+ x_dot (damping=0 -- saf pseudoinverse) ya da
    q_dot = J^T (JJ^T + lambda^2 I)^-1 x_dot (damping>0 -- DLS, singularity
    yakınında daha güvenli).

    cartesian_velocity: J'nin satır sayısına uygun (3, konum-only, ya da 6,
    konum+açısal) bir hız vektörü.
    """
    cartesian_velocity = np.asarray(cartesian_velocity, dtype=float)
    if damping <= 0.0:
        return np.linalg.pinv(J) @ cartesian_velocity

    m = J.shape[0]
    JJt = J @ J.T
    damped = JJt + (damping ** 2) * np.eye(m)
    return J.T @ np.linalg.solve(damped, cartesian_velocity)


def simulate_resolved_rate_motion(
    q_start,
    cartesian_velocity,
    duration: float,
    dt: float = 0.01,
    dh_params=UR5_DH_PARAMS,
    damping: float = 0.05,
    use_orientation: bool = False,
):
    """
    Sabit bir end-effector Cartesian hızını (`cartesian_velocity`, 3 ya da 6
    boyutlu) `duration` saniye boyunca, `dt` adımlarla Euler entegrasyonuyla
    izler. Her adımda: mevcut Jacobian'dan q_dot hesaplanır, q += q_dot*dt.

    Döndürür: (q_history (N, n_joints), ee_position_history (N, 3))
    -- gerçek robotta bu, teleoperation/velocity-control döngüsünün ta kendisidir.
    """
    q = np.array(q_start, dtype=float)
    n_steps = max(int(round(duration / dt)), 1)

    q_history = [q.copy()]
    ee_history = [forward_kinematics(q, dh_params)[:3, 3].copy()]

    for _ in range(n_steps):
        J_full = compute_jacobian(q, dh_params)
        J = J_full if use_orientation else J_full[:3, :]
        q_dot = resolved_rate_joint_velocity(J, cartesian_velocity, damping=damping)
        q = q + q_dot * dt
        q_history.append(q.copy())
        ee_history.append(forward_kinematics(q, dh_params)[:3, 3].copy())

    return np.array(q_history), np.array(ee_history)
