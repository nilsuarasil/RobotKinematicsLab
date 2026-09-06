import numpy as np
import pytest

from src.kinematics.rotations import rot_x, rot_y, rot_z, is_rotation_matrix


def test_identity_at_zero():
    """Sıfır açıda hiçbir rotasyon uygulanmamalı -> birim matris."""
    assert np.allclose(rot_x(0), np.eye(3))
    assert np.allclose(rot_y(0), np.eye(3))
    assert np.allclose(rot_z(0), np.eye(3))


def test_rotations_are_valid_rotation_matrices():
    """Rastgele açılarda üretilen matrisler ortogonal olmalı ve det=+1 vermeli."""
    for theta in [0.1, 1.0, np.pi / 2, np.pi, 3.7]:
        assert is_rotation_matrix(rot_x(theta))
        assert is_rotation_matrix(rot_y(theta))
        assert is_rotation_matrix(rot_z(theta))


def test_rot_z_90_maps_x_axis_to_y_axis():
    """Z ekseni etrafında 90 derece dönünce, X ekseni Y eksenine gitmeli."""
    R = rot_z(np.pi / 2)
    x_axis = np.array([1, 0, 0])
    result = R @ x_axis
    assert np.allclose(result, [0, 1, 0], atol=1e-9)


def test_rot_x_90_maps_y_axis_to_z_axis():
    """X ekseni etrafında 90 derece dönünce, Y ekseni Z eksenine gitmeli."""
    R = rot_x(np.pi / 2)
    y_axis = np.array([0, 1, 0])
    result = R @ y_axis
    assert np.allclose(result, [0, 0, 1], atol=1e-9)


def test_rot_y_90_maps_z_axis_to_x_axis():
    """Y ekseni etrafında 90 derece dönünce, Z ekseni X eksenine gitmeli."""
    R = rot_y(np.pi / 2)
    z_axis = np.array([0, 0, 1])
    result = R @ z_axis
    assert np.allclose(result, [1, 0, 0], atol=1e-9)


def test_composition_is_not_commutative():
    """
    Rotasyon sırası önemlidir: Rz(90) @ Rx(90) genelde
    Rx(90) @ Rz(90)'a eşit değildir.
    """
    a = rot_z(np.pi / 2) @ rot_x(np.pi / 2)
    b = rot_x(np.pi / 2) @ rot_z(np.pi / 2)
    assert not np.allclose(a, b)


def test_is_rotation_matrix_rejects_reflection():
    """Determinantı -1 olan (yansıma içeren) bir matris rotasyon değildir."""
    reflection = np.diag([1, 1, -1]).astype(float)
    assert not is_rotation_matrix(reflection)


def test_is_rotation_matrix_rejects_non_orthogonal():
    """Ortogonal olmayan bir matris rotasyon matrisi olamaz."""
    not_orthogonal = np.array([
        [1, 0.5, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    assert not is_rotation_matrix(not_orthogonal)
