"""
Forward Kinematics: eklem açıları (q1..q6) verilince end-effector pose'unu
hesaplar. DH tablosundaki her satırdan bir transform üretip zincirler:
    T06 = T01 . T12 . T23 . T34 . T45 . T56
"""
import numpy as np
from src.kinematics.dh import dh_transform, UR5_DH_PARAMS


def forward_kinematics_all(q, dh_params=UR5_DH_PARAMS):
    """
    q: 6 eklem açısı (radyan)
    Döndürür: [T01, T02, T03, T04, T05, T06] -- her biri base frame'e göre
    kümülatif transform (Jacobian hesaplarken ara sonuçlar gerekir).
    """
    if len(q) != len(dh_params):
        raise ValueError(f"{len(dh_params)} eklem açısı bekleniyor, {len(q)} verildi")

    transforms = []
    T = np.eye(4)
    for (a, alpha, d), theta in zip(dh_params, q):
        T_i = dh_transform(a, alpha, d, theta)
        T = T @ T_i
        transforms.append(T)
    return transforms


def forward_kinematics(q, dh_params=UR5_DH_PARAMS):
    """Sadece son (base -> end-effector) 4x4 transform matrisini döndürür."""
    return forward_kinematics_all(q, dh_params)[-1]


def end_effector_position(q, dh_params=UR5_DH_PARAMS):
    T06 = forward_kinematics(q, dh_params)
    return T06[:3, 3]
