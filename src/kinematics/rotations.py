"""
Ders 1: 3D Rotasyonlar

Temel eksen rotasyon matrislerini ve bir matrisin geçerli bir rotasyon
matrisi olup olmadığını kontrol eden yardımcı fonksiyonları içerir.

Bkz. docs/lessons/01_rotations.md
"""
import numpy as np


def rot_x(theta: float) -> np.ndarray:
    """X ekseni etrafında theta (radyan) kadar döndüren 3x3 rotasyon matrisi."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def rot_y(theta: float) -> np.ndarray:
    """Y ekseni etrafında theta (radyan) kadar döndüren 3x3 rotasyon matrisi."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def rot_z(theta: float) -> np.ndarray:
    """Z ekseni etrafında theta (radyan) kadar döndüren 3x3 rotasyon matrisi."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def is_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Bir matrisin geçerli bir rotasyon matrisi olup olmadığını kontrol eder:
    - Ortogonal olmalı: R^T @ R == I
    - Determinantı +1 olmalı (determinantı -1 olan ortogonal matrisler
      bir yansıma/reflection içerir, gerçek bir rotasyon değildir)
    """
    R = np.asarray(R)
    if R.shape != (3, 3):
        return False
    orthogonal = np.allclose(R.T @ R, np.eye(3), atol=tol)
    proper_det = np.isclose(np.linalg.det(R), 1.0, atol=tol)
    return bool(orthogonal and proper_det)
