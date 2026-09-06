"""
teleop_control_step() -- pynput/CoppeliaSim I/O'suz, saf kontrol döngüsü
matematiği. Fare girdisinin doğru yöne hareket ettirdiğini, titreme
filtresinin gürültülü/salınımlı girdiyi yumuşattığını ve joint limitlerinin
her zaman korunduğunu doğrular.
"""
import numpy as np
from src.teleoperation.mouse_teleop import teleop_control_step
from src.control.motion_scaling import MotionScaler
from src.control.tremor_filter import TremorFilter
from src.robot.robot_model import UR5Robot
from src.robot.joint_limits import JOINT_LIMITS, check_joint_limits


def _default_robot():
    return UR5Robot()


def test_mouse_right_moves_end_effector_in_positive_x_direction():
    robot = _default_robot()
    scaler = MotionScaler(scale=1.0)
    q = np.array([0.2, -0.6, 0.8, -0.2, 0.5, 0.0])

    p0 = robot.forward(q)[:3, 3]
    q_new = teleop_control_step(
        q, master_delta_px=(50.0, 0.0), pixel_to_mps=0.002, scaler=scaler,
        tremor_filter=None, dt=0.05, robot=robot,
    )
    p1 = robot.forward(q_new)[:3, 3]

    # Tam X yönünde olmasa da (Jacobian yerel), pozitif X bileşeni baskın olmalı.
    assert (p1 - p0)[0] > 0


def test_mouse_up_moves_end_effector_in_positive_z_direction():
    robot = _default_robot()
    scaler = MotionScaler(scale=1.0)
    q = np.array([0.2, -0.6, 0.8, -0.2, 0.5, 0.0])

    p0 = robot.forward(q)[:3, 3]
    # Ekranda "yukarı" hareket -> piksel y AZALIR (negatif dy).
    q_new = teleop_control_step(
        q, master_delta_px=(0.0, -50.0), pixel_to_mps=0.002, scaler=scaler,
        tremor_filter=None, dt=0.05, robot=robot,
    )
    p1 = robot.forward(q_new)[:3, 3]
    assert (p1 - p0)[2] > 0


def test_no_mouse_movement_means_no_joint_change():
    robot = _default_robot()
    scaler = MotionScaler(scale=1.0)
    q = np.array([0.2, -0.6, 0.8, -0.2, 0.5, 0.0])

    q_new = teleop_control_step(
        q, master_delta_px=(0.0, 0.0), pixel_to_mps=0.002, scaler=scaler,
        tremor_filter=None, dt=0.05, robot=robot,
    )
    assert np.allclose(q_new, q, atol=1e-9)


def test_motion_scaling_reduces_movement_magnitude():
    robot = _default_robot()
    q = np.array([0.2, -0.6, 0.8, -0.2, 0.5, 0.0])

    q_full = teleop_control_step(
        q, master_delta_px=(50.0, 0.0), pixel_to_mps=0.002, scaler=MotionScaler(scale=1.0),
        tremor_filter=None, dt=0.05, robot=robot,
    )
    q_scaled = teleop_control_step(
        q, master_delta_px=(50.0, 0.0), pixel_to_mps=0.002, scaler=MotionScaler(scale=0.2),
        tremor_filter=None, dt=0.05, robot=robot,
    )
    delta_full = np.linalg.norm(q_full - q)
    delta_scaled = np.linalg.norm(q_scaled - q)
    assert delta_scaled < delta_full
    assert np.isclose(delta_scaled / delta_full, 0.2, atol=0.05)


def test_joint_limits_always_respected_even_with_huge_input():
    robot = _default_robot()
    scaler = MotionScaler(scale=1.0)
    q = np.zeros(6)

    q_new = teleop_control_step(
        q, master_delta_px=(1e6, 1e6), pixel_to_mps=0.002, scaler=scaler,
        tremor_filter=None, dt=0.05, robot=robot, joint_limits=JOINT_LIMITS,
    )
    assert check_joint_limits(q_new, JOINT_LIMITS) == []


def test_tremor_filter_smooths_oscillating_input():
    """Sağa-sola salınan (titreme benzeri) fare girdisiyle beslenen kontrol
    döngüsünde, titreme filtresi AÇIKKEN toplam eklem hareketi (yol uzunluğu),
    KAPALIYKEN olduğundan belirgin şekilde daha az olmalı (yüksek frekanslı
    ileri-geri hareketi büyük ölçüde söndürüyor)."""
    robot = _default_robot()
    scaler = MotionScaler(scale=1.0)
    dt = 0.02  # 50 Hz
    oscillating_deltas = [(30.0 if i % 2 == 0 else -30.0, 0.0) for i in range(40)]

    def run(tremor_filter):
        q = np.array([0.2, -0.6, 0.8, -0.2, 0.5, 0.0])
        path_length = 0.0
        for delta in oscillating_deltas:
            q_new = teleop_control_step(
                q, delta, pixel_to_mps=0.002, scaler=scaler,
                tremor_filter=tremor_filter, dt=dt, robot=robot,
            )
            path_length += np.linalg.norm(q_new - q)
            q = q_new
        return path_length

    path_without_filter = run(None)
    path_with_filter = run(TremorFilter(cutoff_hz=2.0, sample_rate_hz=1.0 / dt, n_channels=3))
    assert path_with_filter < 0.3 * path_without_filter
