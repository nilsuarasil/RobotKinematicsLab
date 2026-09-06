"""
Orientation trajectory: pozisyon quintic ile düz giderken, yönelimin de
YUMUŞAK (ve açısal olarak sabit hızlı) değişmesi için quaternion SLERP
kullanır (bkz. kinematics/orientation.py'deki quaternion_slerp -- asıl
matematik orada, burada sadece trajectory üretimi için sarmalanıyor).

Pipeline (V2 mimarisi notlarındaki gibi):
    Position    -> quintic_scaling_profile (cartesian_trajectory.py)
    Orientation -> quaternion SLERP (bu modül)
    ikisi birlikte -> 6D Cartesian trajectory (cartesian_pose_trajectory)
"""
import numpy as np
from src.planning.quintic import quintic_scaling_profile
from src.kinematics.orientation import (
    quaternion_slerp, rotation_matrix_to_quaternion, quaternion_to_rotation_matrix,
)


def orientation_trajectory(q_start, q_end, steps: int = 50):
    """(steps, 4) şeklinde, quintic zaman ölçekli quaternion SLERP trajectory.
    q_start/q_end: birim quaternion [w,x,y,z]."""
    s = quintic_scaling_profile(steps)
    return np.array([quaternion_slerp(q_start, q_end, si) for si in s])


def cartesian_pose_trajectory(p_start, p_end, R_start, R_end, steps: int = 50):
    """
    Pozisyon (quintic, düz çizgi) ve yönelimi (SLERP) TEK bir zaman
    profiliyle (aynı s(t)) birleştirip, her adım için tam bir 4x4
    homogeneous transform listesi döndürür -- "6D Cartesian trajectory".
    """
    from src.kinematics.transforms import homogeneous_transform

    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    q_start = rotation_matrix_to_quaternion(R_start)
    q_end = rotation_matrix_to_quaternion(R_end)

    s = quintic_scaling_profile(steps)
    poses = []
    for si in s:
        p = p_start + si * (p_end - p_start)
        q = quaternion_slerp(q_start, q_end, si)
        R = quaternion_to_rotation_matrix(q)
        poses.append(homogeneous_transform(R, p))
    return poses
