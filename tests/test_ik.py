import numpy as np
from src.kinematics.inverse import inverse_kinematics
from src.kinematics.forward import end_effector_position


def test_ik_converges_to_reachable_target():
    # Bilinen bir q'dan FK ile ulaşılabilir bir hedef üretiyoruz, böylece
    # IK'nın "doğru" cevaba yaklaşıp yaklaşmadığını (aynı q olması şart
    # değil, IK çözümleri tekil olmayabilir -- ama FK(q_ik) ~= target olmalı)
    # kontrol edebiliyoruz.
    q_true = [0.3, -0.5, 0.6, 0.1, -0.2, 0.4]
    target = end_effector_position(q_true)

    result = inverse_kinematics(target, q_init=[0, 0, 0, 0, 0, 0])

    assert result.converged
    assert result.final_error < 1e-3

    achieved = end_effector_position(result.q)
    assert np.linalg.norm(achieved - np.asarray(target)) < 1e-3


def test_ik_reports_unconverged_for_unreachable_target():
    result = inverse_kinematics([4.0, 0.0, 0.0], max_iters=100)
    assert not result.converged
    assert result.final_error > 1e-3


def test_ik_result_has_correct_number_of_joints():
    result = inverse_kinematics([0.3, 0.1, 0.4])
    assert len(result.q) == 6


def test_ik_flags_joint_limit_violations_when_present():
    # Çok kısıtlı bir limit seti vererek (neredeyse tüm çözümleri reddeden)
    # violation mekanizmasının çalıştığını doğruluyoruz.
    tiny_limits = [(-0.01, 0.01)] * 6
    result = inverse_kinematics([0.3, 0.1, 0.4], joint_limits=tiny_limits, max_iters=50)
    assert len(result.joint_limit_violations) > 0
