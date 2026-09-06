import numpy as np
from src.robot.robot_model import UR5Robot


def test_end_effector_position_shifts_with_base():
    robot_origin = UR5Robot(base_position=[0, 0, 0])
    robot_shifted = UR5Robot(base_position=[1, 0, 0])

    q = [0.2, -0.3, 0.4, 0.1, -0.1, 0.2]
    p0 = robot_origin.end_effector_position(q)
    p1 = robot_shifted.end_effector_position(q)

    assert np.allclose(p1 - p0, [1, 0, 0], atol=1e-9)


def test_solve_ik_round_trip_with_base_offset():
    robot = UR5Robot(base_position=[0.5, 0.0, 0.0])
    q_true = [0.3, -0.4, 0.5, 0.2, -0.2, 0.1]
    target_world = robot.end_effector_position(q_true)

    result = robot.solve_ik(target_world, q_init=[0, 0, 0, 0, 0, 0])
    assert result.converged

    achieved_world = robot.end_effector_position(result.q)
    assert np.linalg.norm(achieved_world - target_world) < 1e-3


def test_is_reachable_uses_base_position():
    robot = UR5Robot(base_position=[0.5, 0.0, 0.0])
    reachable, dist = robot.is_reachable([0.9, 0.0, 0.0])
    assert reachable
    assert np.isclose(dist, 0.4)


def test_solve_ik_round_trip_with_rotated_base():
    """
    CoppeliaSim sahnesinde robot sadece taşınmış değil, döndürülmüş de
    olabilir. Bu test, taban rotasyonu sıfır olmadığında bile IK'nın hâlâ
    doğru bir dünya-çerçevesi hedefine yakınsadığını doğruluyor -- yani
    "sadece öteleme" değil, tam transform (rotasyon dahil) doğru çalışıyor.
    """
    from src.kinematics.rotations import rot_z
    base_rotation = rot_z(np.pi / 2)  # taban dünya çerçevesine göre 90 derece dönük
    robot = UR5Robot(base_position=[0.2, -0.3, 0.0], base_rotation=base_rotation)

    q_true = [0.4, -0.2, 0.6, 0.1, 0.3, -0.1]
    target_world = robot.end_effector_position(q_true)

    result = robot.solve_ik(target_world, q_init=[0, 0, 0, 0, 0, 0])
    assert result.converged

    achieved_world = robot.end_effector_position(result.q)
    assert np.linalg.norm(achieved_world - target_world) < 1e-3


def test_base_transform_overrides_position_and_rotation():
    from src.kinematics.rotations import rot_z
    from src.kinematics.transforms import homogeneous_transform

    T = homogeneous_transform(rot_z(0.5), [1.0, 2.0, 3.0])
    robot = UR5Robot(base_transform=T)

    assert np.allclose(robot.base_position, [1.0, 2.0, 3.0])
    assert np.allclose(robot.base_transform, T)


def test_solve_ik_v2_position_only_with_rotated_base():
    from src.kinematics.rotations import rot_z
    base_rotation = rot_z(np.pi / 3)
    robot = UR5Robot(base_position=[0.2, -0.1, 0.0], base_rotation=base_rotation)

    q_true = [0.3, -0.4, 0.5, 0.2, -0.2, 0.1]
    target_world = robot.end_effector_position(q_true)

    for solver in ("pseudoinverse", "dls", "transpose"):
        result = robot.solve_ik_v2(target_world, q_init=np.zeros(6), solver=solver, max_iters=1000)
        assert result.converged, f"{solver} yakınsamadı"
        achieved_world = robot.forward(result.q)[:3, 3]
        assert np.linalg.norm(achieved_world - target_world) < 1e-3


def test_solve_ik_v2_with_orientation_target_world_frame():
    from src.kinematics.rotations import rot_z
    base_rotation = rot_z(np.pi / 4)
    robot = UR5Robot(base_position=[0.1, 0.1, 0.0], base_rotation=base_rotation)

    q_true = [0.2, -0.3, 0.4, 0.1, 0.5, -0.2]
    T_true_world = robot.forward(q_true)
    target_position_world = T_true_world[:3, 3]
    target_R_world = T_true_world[:3, :3]

    result = robot.solve_ik_v2(
        target_position_world, target_orientation=target_R_world,
        q_init=np.array([0.1] * 6), solver="dls", max_iters=2000,
        pos_tol=1e-3, ori_tol=5e-3,
    )
    assert result.converged
    T_achieved_world = robot.forward(result.q)
    assert np.linalg.norm(T_achieved_world[:3, 3] - target_position_world) < 1e-3


def test_solve_ik_v2_multi_start_returns_selected_candidate():
    robot = UR5Robot(base_position=[0.3, 0.0, 0.0])
    q_true = [0.3, -0.4, 0.5, 0.2, -0.2, 0.1]
    target_world = robot.end_effector_position(q_true)

    multi_result = robot.solve_ik_v2(
        target_world, q_init=np.zeros(6), solver="dls",
        use_multi_start=True, n_random_seeds=3,
    )
    assert multi_result.selected.converged
    achieved_world = robot.forward(multi_result.selected.q)[:3, 3]
    assert np.linalg.norm(achieved_world - target_world) < 1e-3
