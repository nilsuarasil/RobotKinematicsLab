import numpy as np
from src.kinematics.jacobian import compute_jacobian, manipulability, is_near_singularity
from src.kinematics.forward import end_effector_position


def numerical_position_jacobian(q, eps=1e-6):
    q = np.array(q, dtype=float)
    n = len(q)
    J_num = np.zeros((3, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        p_plus = end_effector_position(q + dq)
        p_minus = end_effector_position(q - dq)
        J_num[:, i] = (p_plus - p_minus) / (2 * eps)
    return J_num


def test_jacobian_position_rows_match_numerical():
    q = [0.3, -0.4, 0.5, 0.2, -0.3, 0.1]
    J = compute_jacobian(q)
    J_num = numerical_position_jacobian(q)
    assert np.allclose(J[:3, :], J_num, atol=1e-4)


def test_jacobian_shape():
    J = compute_jacobian([0, 0, 0, 0, 0, 0])
    assert J.shape == (6, 6)


def test_manipulability_nonnegative():
    J = compute_jacobian([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert manipulability(J) >= 0


def test_is_near_singularity_runs():
    result = is_near_singularity([0, 0, 0, 0, 0, 0])
    assert isinstance(result, (bool, np.bool_))
