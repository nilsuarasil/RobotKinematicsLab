"""
ik_pseudoinverse.py / ik_dls.py / ik_jacobian_transpose.py arasında paylaşılan
yardımcı fonksiyonlar: hedef orientation'ı normalize etme, 6D (ya da
orientation verilmediyse 3D) hata vektörü hesaplama.

Üç solver'ın da ana döngüsü kasıtlı olarak KENDİ dosyasında, okunabilir
şekilde tekrar yazılıyor (sadece dq = ... satırı farklı) -- amaç, her
solver'ın matematiğinin tek bir dosyaya bakarak anlaşılabilmesi.
"""
import numpy as np
from src.kinematics.orientation import (
    rpy_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    orientation_error,
)


def resolve_target_rotation(target_orientation):
    """
    target_orientation: None, 3x3 rotasyon matrisi, quaternion (len 4, [w,x,y,z])
    veya RPY (len 3, [roll, pitch, yaw] radyan) olabilir. Hepsini 3x3 rotasyon
    matrisine (ya da hedeflenmediyse None'a) çevirir.
    """
    if target_orientation is None:
        return None
    arr = np.asarray(target_orientation, dtype=float)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (4,):
        return quaternion_to_rotation_matrix(arr)
    if arr.shape == (3,):
        return rpy_to_rotation_matrix(arr[0], arr[1], arr[2])
    raise ValueError(
        "target_orientation 3x3 matris, 4'lü quaternion ya da 3'lü RPY olmalı, "
        f"alınan şekil: {arr.shape}"
    )


def pose_error_vector(T_current, target_position, target_R):
    """
    Döndürür: (error_vector, position_error_norm, orientation_error_norm)
    target_R None ise error_vector 3 boyutludur (sadece pozisyon); değilse
    6 boyutludur ([pos(3), orientation(3)]).
    """
    target_position = np.asarray(target_position, dtype=float)
    pos_err = target_position - T_current[:3, 3]
    pos_err_norm = float(np.linalg.norm(pos_err))

    if target_R is None:
        return pos_err, pos_err_norm, 0.0

    ori_err = orientation_error(T_current[:3, :3], target_R)
    ori_err_norm = float(np.linalg.norm(ori_err))
    return np.concatenate([pos_err, ori_err]), pos_err_norm, ori_err_norm


def select_jacobian_rows(J_full, use_orientation: bool):
    """6xN geometric Jacobian'dan, hedefe göre 3xN (sadece pozisyon) ya da
    6xN (pozisyon+orientation) satırları seçer."""
    return J_full if use_orientation else J_full[:3, :]


def wrap_to_pi(q):
    return (q + np.pi) % (2 * np.pi) - np.pi
