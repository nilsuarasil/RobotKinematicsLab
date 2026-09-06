import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.jacobian import compute_jacobian
from src.analysis.singularity import (
    singular_values, min_singular_value, condition_number, determinant_measure,
    analyze_singularity,
)


def test_singular_values_sorted_descending():
    J = compute_jacobian(np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1]), UR5_DH_PARAMS)
    svals = singular_values(J)
    assert list(svals) == sorted(svals, reverse=True)


def test_determinant_measure_none_for_nonsquare():
    J = np.zeros((3, 6))
    assert determinant_measure(J) is None


def test_known_singular_configuration_detected():
    """
    UR5 icin q = [0,0,0,0,0,0] noktasinda J5=J4 gibi bazi eksenler hizalanabilir;
    burada daha guvenilir bir test icin, KENDI kurdugumuz, bilinen sekilde
    tekil bir Jacobian (rank-deficient) matrisiyle dogrudan test ediyoruz.
    """
    # Rank-deficient (satirlar lineer bagimli -> JJ^T tekil): 3. satir 1. satirin kopyasi.
    J = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    report = analyze_singularity(J, sigma_threshold=1e-3)
    assert report.min_singular_value < 1e-6
    assert report.status == "SINGULAR"
    assert report.is_singular


def test_well_conditioned_jacobian_is_safe():
    J = np.eye(6)
    report = analyze_singularity(J)
    assert report.status == "SAFE"
    assert not report.is_singular
    assert np.isclose(report.condition_number, 1.0)


def test_condition_number_matches_manual_ratio():
    J = np.diag([2.0, 2.0, 2.0, 1.0, 1.0, 0.5])
    kappa = condition_number(J)
    assert np.isclose(kappa, 2.0 / 0.5)
