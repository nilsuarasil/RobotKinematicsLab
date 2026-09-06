import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.jacobian import compute_jacobian, manipulability as legacy_manipulability
from src.analysis.manipulability import (
    yoshikawa_manipulability, manipulability_from_singular_values,
)


def test_yoshikawa_matches_svd_product():
    rng = np.random.default_rng(3)
    q = rng.uniform(-np.pi, np.pi, size=6)
    J = compute_jacobian(q, UR5_DH_PARAMS)
    w1 = yoshikawa_manipulability(J)
    w2 = manipulability_from_singular_values(J)
    assert np.isclose(w1, w2, atol=1e-8)


def test_yoshikawa_matches_legacy_jacobian_module_version():
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    J = compute_jacobian(q, UR5_DH_PARAMS)
    assert np.isclose(yoshikawa_manipulability(J), legacy_manipulability(J), atol=1e-8)


def test_manipulability_positive_for_generic_config():
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    J = compute_jacobian(q, UR5_DH_PARAMS)
    assert yoshikawa_manipulability(J) > 0.0


def test_manipulability_zero_for_rank_deficient_jacobian():
    # Satirlar lineer bagimli (3. satir = 1. satir) -> J tam satir rankina
    # sahip degil -> JJ^T tekil -> manipulability = 0.
    J = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    assert np.isclose(yoshikawa_manipulability(J), 0.0, atol=1e-8)
