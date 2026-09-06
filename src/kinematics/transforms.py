"""
Homogeneous Transformation Matrices

4x4 matrisler ile rotasyon + öteleme (translation) birlikte ifade edilir:

    T = [ R   t ]
        [ 0   1 ]
"""
import numpy as np


def homogeneous_transform(R: np.ndarray, t) -> np.ndarray:
    """3x3 rotasyon matrisi R ve 3'lük öteleme vektörü t'den 4x4 T matrisi oluşturur."""
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_point(T: np.ndarray, p) -> np.ndarray:
    """4x4 T matrisini bir 3D noktaya uygular."""
    p = np.asarray(p, dtype=float).reshape(3)
    p_h = np.append(p, 1.0)
    result = T @ p_h
    return result[:3]


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """
    Bir homogeneous transform'un tersini alır:
        T^-1 = [ R^T   -R^T @ t ]
               [  0        1    ]
    (Rotasyon matrisi ortogonal olduğu için genel 4x4 matris tersi almaktan
    çok daha ucuz bir hesap.)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return homogeneous_transform(R_inv, t_inv)


def position_from_transform(T: np.ndarray) -> np.ndarray:
    return T[:3, 3].copy()


def rotation_from_transform(T: np.ndarray) -> np.ndarray:
    return T[:3, :3].copy()
