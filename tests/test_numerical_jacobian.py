"""
Analitik geometric Jacobian'ı (jacobian.py) bağımsız bir sayısal (finite
difference) yöntemle (numerical_jacobian.py) doğrular. Bu, "matematiği doğru
uyguladığımızı" gösteren en önemli testlerden biri -- iki farklı yöntem
(analitik formül vs. sayısal türev) aynı sonucu vermeli.
"""
import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.kinematics.jacobian import compute_geometric_jacobian
from src.kinematics.numerical_jacobian import numerical_geometric_jacobian, skew, vee


def test_skew_vee_are_inverses():
    v = np.array([0.3, -0.7, 1.2])
    assert np.allclose(vee(skew(v)), v, atol=1e-12)


def test_numerical_matches_analytical_jacobian_at_random_configs():
    rng = np.random.default_rng(42)
    fk = lambda q: forward_kinematics(q, UR5_DH_PARAMS)

    for _ in range(10):
        q = rng.uniform(-np.pi, np.pi, size=6)
        J_analytic = compute_geometric_jacobian(q, UR5_DH_PARAMS)
        J_numeric = numerical_geometric_jacobian(fk, q, eps=1e-6)
        max_diff = np.max(np.abs(J_analytic - J_numeric))
        assert max_diff < 1e-4, f"max diff too large: {max_diff}"


def test_numerical_matches_analytical_jacobian_at_zero_pose():
    q = np.zeros(6)
    fk = lambda qq: forward_kinematics(qq, UR5_DH_PARAMS)
    J_analytic = compute_geometric_jacobian(q, UR5_DH_PARAMS)
    J_numeric = numerical_geometric_jacobian(fk, q, eps=1e-6)
    assert np.allclose(J_analytic, J_numeric, atol=1e-4)
