"""
CalibratedUR5Robot, CoppeliaSim sahnesinden EMPİRİK olarak okunan (rastgele
görünebilecek) montaj transformlarıyla çalışmalı -- DH tablosuna bağlı
DEĞİL. Bu yüzden burada gerçek UR5 geometrisi yerine, rastgele ama geçerli
(rijit) transformlardan oluşan sentetik bir "sahne" kurup, sınıfın zincirleme
matematiğinin (T = base . T0.Rz(q0) . T1.Rz(q1) ... . tip_offset) doğru
çalıştığını ve genel sayısal IK'nın böyle bir zincir için de yakınsadığını
doğruluyoruz.
"""
import numpy as np
from src.kinematics.rotations import rot_x, rot_y, rot_z
from src.kinematics.transforms import homogeneous_transform
from src.robot.robot_model import CalibratedUR5Robot


def _random_static_transform(rng):
    rx, ry, rz = rng.uniform(-0.6, 0.6, size=3)
    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    t = rng.uniform(-0.3, 0.3, size=3)
    return homogeneous_transform(R, t)


def _ground_truth_forward(base_transform, joint_local_transforms, tip_offset, q):
    T = base_transform
    for T_local, theta in zip(joint_local_transforms, q):
        T = T @ T_local @ homogeneous_transform(rot_z(theta), [0, 0, 0])
    return T @ tip_offset


def _random_calibration(rng):
    return {
        "base_transform": _random_static_transform(rng),
        "joint_local_transforms": [_random_static_transform(rng) for _ in range(6)],
        "tip_offset": _random_static_transform(rng),
    }


def test_calibrated_robot_matches_ground_truth_chain():
    rng = np.random.default_rng(42)
    calibration = _random_calibration(rng)
    robot = CalibratedUR5Robot(calibration)

    q = rng.uniform(-1.0, 1.0, size=6)
    expected = _ground_truth_forward(
        calibration["base_transform"], calibration["joint_local_transforms"],
        calibration["tip_offset"], q,
    )[:3, 3]
    actual = robot.end_effector_position(q)
    assert np.allclose(actual, expected, atol=1e-9)


def test_calibrated_robot_solve_ik_round_trip():
    rng = np.random.default_rng(7)
    calibration = _random_calibration(rng)
    robot = CalibratedUR5Robot(calibration)

    q_true = rng.uniform(-1.0, 1.0, size=6)
    target = robot.end_effector_position(q_true)

    result = robot.solve_ik(target, q_init=np.zeros(6))
    assert result.converged

    achieved = robot.end_effector_position(result.q)
    assert np.linalg.norm(achieved - target) < 1e-3


def test_calibrated_robot_is_reachable_uses_base_position():
    calibration = {
        "base_transform": homogeneous_transform(np.eye(3), [1.0, 0.0, 0.0]),
        "joint_local_transforms": [np.eye(4)] * 6,
        "tip_offset": np.eye(4),
    }
    robot = CalibratedUR5Robot(calibration)
    reachable, dist = robot.is_reachable([1.4, 0.0, 0.0])
    assert reachable
    assert np.isclose(dist, 0.4)


def test_calibrated_robot_identity_calibration_matches_dh_zero_pose_shape():
    """
    Kalibrasyon transformları hepsi birim (identity) ise, robot her zaman
    orijinde durur -- bu en azından "hiçbir şey patlamıyor" temel sağlamasını
    yapıyor.
    """
    calibration = {
        "base_transform": np.eye(4),
        "joint_local_transforms": [np.eye(4)] * 6,
        "tip_offset": np.eye(4),
    }
    robot = CalibratedUR5Robot(calibration)
    pos = robot.end_effector_position([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert np.allclose(pos, [0, 0, 0])
