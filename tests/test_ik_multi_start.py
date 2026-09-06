import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.kinematics.ik_multi_start import multi_start_ik, generate_seed_configs


def test_generate_seed_configs_returns_labeled_seeds():
    seeds = generate_seed_configs(n_joints=6, n_random=4, seed=1)
    assert len(seeds) == 3 + 4
    labels = [s[0] for s in seeds]
    assert "zero" in labels
    for _, q in seeds:
        assert len(q) == 6


def test_multi_start_ik_finds_converged_candidate():
    q_true = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    result = multi_start_ik(target_position=target, q_current=np.zeros(6), n_random_seeds=3)
    assert result.selected.converged
    assert result.selected.pose_error_mm < 1.0
    assert len(result.candidates) == 3 + 3


def test_multi_start_ik_prefers_lower_cost_among_converged():
    q_true = np.array([0.2, -0.3, 0.4, 0.1, 0.5, -0.2])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    result = multi_start_ik(target_position=target, q_current=np.zeros(6), n_random_seeds=4, seed=7)
    converged = [c for c in result.candidates if c.converged]
    assert any(result.selected is c for c in converged)
    assert result.selected.cost == min(c.cost for c in converged)
