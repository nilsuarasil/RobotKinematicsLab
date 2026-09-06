"""Hedef ile gerçekleşen (actual) end-effector pose'u arasındaki hata hesapları."""
import numpy as np
from src.kinematics.orientation import orientation_error, quaternion_error_norm


def position_error(actual, target) -> float:
    """Öklidyen mesafe hatası (metre cinsinden)."""
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.linalg.norm(actual - target))


def position_error_mm(actual, target) -> float:
    return position_error(actual, target) * 1000.0


def orientation_error_deg(R_actual, R_target) -> float:
    """İki rotasyon matrisi arasındaki açısal hata (derece cinsinden)."""
    return float(np.degrees(quaternion_error_norm(R_actual, R_target)))


def pose_error_6d(T_actual, T_target):
    """
    4x4 homogeneous transform'lardan tam 6D pose hatasını çıkarır.
    Döndürür: (position_error_m, orientation_error_rad, error_vector_6)
    error_vector_6 = [ex, ey, ez, e_wx, e_wy, e_wz] -- IK'nın kullandığı
    hata vektörüyle birebir aynı (bkz. ik_dls.py / ik_pseudoinverse.py).
    """
    T_actual = np.asarray(T_actual, dtype=float)
    T_target = np.asarray(T_target, dtype=float)

    pos_err_vec = T_target[:3, 3] - T_actual[:3, 3]
    ori_err_vec = orientation_error(T_actual[:3, :3], T_target[:3, :3])

    error_vector = np.concatenate([pos_err_vec, ori_err_vec])
    return float(np.linalg.norm(pos_err_vec)), float(np.linalg.norm(ori_err_vec)), error_vector
