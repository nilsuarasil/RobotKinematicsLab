"""
Geometric Jacobian (6x6): eklem hızları ile end-effector doğrusal+açısal
hızı arasındaki ilişkiyi kurar:  ẋ = J(q) q̇

Her (revolute) eklem için:
    J_v_i = z_{i-1} x (o_n - o_{i-1})
    J_w_i = z_{i-1}

Burada z_{i-1}, i. eklemden önceki frame'in Z ekseni (dönme ekseni);
o_{i-1} o frame'in orijini; o_n ise end-effector'ün orijinidir.
"""
import numpy as np
from src.kinematics.forward import forward_kinematics_all
from src.kinematics.dh import UR5_DH_PARAMS


def compute_jacobian(q, dh_params=UR5_DH_PARAMS):
    n = len(dh_params)
    transforms = forward_kinematics_all(q, dh_params)

    z_prev = [np.array([0.0, 0.0, 1.0])]   # frame 0 (base) Z ekseni
    o_prev = [np.array([0.0, 0.0, 0.0])]   # frame 0 (base) orijini
    for T in transforms:
        z_prev.append(T[:3, 2])
        o_prev.append(T[:3, 3])

    o_n = o_prev[-1]  # end-effector orijini

    J = np.zeros((6, n))
    for i in range(n):
        z_i_minus_1 = z_prev[i]
        o_i_minus_1 = o_prev[i]
        J[:3, i] = np.cross(z_i_minus_1, o_n - o_i_minus_1)
        J[3:, i] = z_i_minus_1
    return J


def compute_geometric_jacobian(q, dh_params=UR5_DH_PARAMS):
    """compute_jacobian ile aynı (6xN geometric Jacobian) -- V2 modüllerinde
    isim netliği için (translational/rotational ile birlikte) bu ad tercih
    ediliyor."""
    return compute_jacobian(q, dh_params)


def compute_translational_jacobian(q, dh_params=UR5_DH_PARAMS):
    """Geometric Jacobian'ın ilk 3 satırı: J_v (eklem hızı -> doğrusal hız)."""
    return compute_jacobian(q, dh_params)[:3, :]


def compute_rotational_jacobian(q, dh_params=UR5_DH_PARAMS):
    """Geometric Jacobian'ın son 3 satırı: J_w (eklem hızı -> açısal hız)."""
    return compute_jacobian(q, dh_params)[3:, :]


def manipulability(J):
    """Yoshikawa manipulability ölçüsü: sqrt(det(J J^T)). 0'a yaklaşması
    singularity'ye yaklaşıldığının işaretidir."""
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(det, 0.0)))


def is_near_singularity(q, dh_params=UR5_DH_PARAMS, threshold=1e-3):
    J = compute_jacobian(q, dh_params)
    return manipulability(J) < threshold
