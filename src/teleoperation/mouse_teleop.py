"""
CoppeliaSim'de gerçek zamanlı fare (mouse) teleoperation.

Pipeline (proje mimarisi notlarındaki gibi):

    Mouse hareketi (piksel)
        -> Motion Scaling (control/motion_scaling.py)
        -> Tremor Filter (control/tremor_filter.py, opsiyonel)
        -> Hedef Cartesian hız
        -> Resolved-Rate IK (control/resolved_rate.py, J^+ ile)
        -> Eklem hızı -> entegre edilip (q += q_dot*dt) hedef eklem açısı
        -> CoppeliaSim'e pozisyon komutu (CoppeliaSimClient.set_joint_positions)

Robotun GERÇEK geometrisi -- main.py'deki --sim modunda olduğu gibi --
CoppeliaSim sahnesinden empirik olarak kalibre edilir (CalibratedUR5Robot).
Bu robotun analitik bir Jacobian'ı olmadığı için (geometri koddan değil
sahneden okunuyor), her kontrol adımında Jacobian'ı `numerical_jacobian.py`
ile (merkezi fark) sayısal olarak hesaplıyoruz -- 6 eklem için bu ucuz bir
işlemdir (12 forward-kinematics değerlendirmesi), 20-50 Hz kontrol döngüsü
için sorun teşkil etmez.

ÖNEMLİ -- bu modül CoppeliaSim'in GUI'siyle ve gerçek fare girdisiyle
etkileşime girdiği için (pynput gerektirir), bulut sandbox'ında canlı olarak
test EDİLEMEDİ. Kontrol döngüsünün matematiği (`teleop_control_step`) ayrı,
saf (yan etkisiz) bir fonksiyon olarak yazıldı ve tam test kapsamı bununla
sağlandı (bkz. tests/test_mouse_teleop.py); pynput/CoppeliaSim I/O katmanı
(`run_mouse_teleoperation`) senin makinende ilk çalıştırmada küçük bir
ayar/debug turu gerektirebilir -- coppelia_client.py'de olduğu gibi.
"""
import time
import threading
import numpy as np

from src.kinematics.numerical_jacobian import numerical_geometric_jacobian
from src.control.resolved_rate import resolved_rate_joint_velocity
from src.control.motion_scaling import MotionScaler, SCALE_PRESETS
from src.control.tremor_filter import TremorFilter
from src.robot.joint_limits import JOINT_LIMITS, clamp_to_limits

try:
    from pynput import mouse, keyboard
except ImportError:
    mouse = None
    keyboard = None


def teleop_control_step(q, master_delta_px, pixel_to_mps: float, scaler: MotionScaler,
                         tremor_filter, dt: float, robot, damping: float = 0.05,
                         joint_limits=JOINT_LIMITS):
    """
    Teleoperation kontrol döngüsünün TEK bir adımını, yan etkisiz (I/O
    içermeyen) bir fonksiyon olarak uygular -- pynput/CoppeliaSim'den
    bağımsız olduğu için doğrudan birim testi yazılabilir.

    master_delta_px: (dx, dy) -- son kontrol adımından bu yana biriken fare
    hareketi (piksel). Eksen eşlemesi: dx -> dünya X (sağ/sol), -dy -> dünya
    Z (yukarı/aşağı; ekran Y'si aşağı doğru arttığı için ters çevrilir).
    Derinlik (dünya Y) şimdilik sabit -- klavye/scroll ile genişletilebilir.

    robot: `.forward(q) -> 4x4 T` metoduna sahip herhangi bir robot modeli
    (UR5Robot ya da CalibratedUR5Robot).

    Döndürür: yeni q (joint limitlerine clip edilmiş).
    """
    dx_px, dy_px = master_delta_px
    master_delta = np.array([dx_px, 0.0, -dy_px], dtype=float) * pixel_to_mps
    robot_delta = scaler.apply(master_delta)

    if tremor_filter is not None:
        robot_delta = tremor_filter.filter_sample(robot_delta)

    cartesian_velocity = robot_delta / dt

    J_full = numerical_geometric_jacobian(robot.forward, np.asarray(q, dtype=float), eps=1e-6)
    J = J_full[:3, :]
    q_dot = resolved_rate_joint_velocity(J, cartesian_velocity, damping=damping)

    q_new = np.asarray(q, dtype=float) + q_dot * dt
    return np.array(clamp_to_limits(q_new, joint_limits))


class MouseInputSource:
    """pynput ile fare hareketini arka planda dinler; ana kontrol döngüsü her
    tick'te `get_delta_and_reset()` çağırarak son tick'ten bu yana biriken
    (dx, dy) piksel hareketini alır (mutlak konum değil, delta -- böylece
    fare ekranın neresinde olursa olsun aynı şekilde çalışır)."""

    def __init__(self):
        if mouse is None:
            raise ImportError("pynput kurulu değil. Kur: pip install pynput")
        self._lock = threading.Lock()
        self._dx_accum = 0.0
        self._dy_accum = 0.0
        self._last_pos = None
        self._listener = mouse.Listener(on_move=self._on_move)

    def _on_move(self, x, y):
        with self._lock:
            if self._last_pos is not None:
                self._dx_accum += x - self._last_pos[0]
                self._dy_accum += y - self._last_pos[1]
            self._last_pos = (x, y)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def get_delta_and_reset(self):
        with self._lock:
            dx, dy = self._dx_accum, self._dy_accum
            self._dx_accum = 0.0
            self._dy_accum = 0.0
        return dx, dy


class QuitOnEscape:
    """ESC tuşuna basılınca `should_quit=True` olan basit bir klavye dinleyici."""

    def __init__(self):
        if keyboard is None:
            raise ImportError("pynput kurulu değil. Kur: pip install pynput")
        self.should_quit = False
        self._listener = keyboard.Listener(on_press=self._on_press)

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            self.should_quit = True
            return False  # dinleyiciyi durdur

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()


def run_mouse_teleoperation(
    scale_label: str = "5:1",
    pixel_to_mps: float = 0.002,
    tremor_filter_enabled: bool = True,
    tremor_cutoff_hz: float = 2.0,
    control_rate_hz: float = 20.0,
    damping: float = 0.05,
    joint_limits=JOINT_LIMITS,
):
    """
    CoppeliaSim'e bağlanır, sahneyi (q=0 iken) kalibre eder, ardından ESC'ye
    basılana ya da Ctrl+C'ye kadar fareyi dinleyip robotu gerçek zamanlı
    sürükler. `python -m src.teleoperation.mouse_teleop` ile çalıştırılır.

    Ölçek etiketleri: bkz. control/motion_scaling.SCALE_PRESETS ("1:1",
    "2:1", "5:1", "10:1"). Titreme filtresi kapatılırsa (tremor_filter_enabled
    =False), ham (filtresiz) fare hareketi doğrudan uygulanır -- karşılaştırma
    için kullanışlı.
    """
    if mouse is None:
        raise ImportError(
            "pynput kurulu değil. Bu özellik gerçek zamanlı fare girdisi "
            "gerektirir. Kur: pip install pynput"
        )

    from src.simulation.coppelia_client import CoppeliaSimClient
    from src.robot.robot_model import CalibratedUR5Robot

    dt = 1.0 / control_rate_hz
    scaler = MotionScaler.from_preset(scale_label)
    tremor_filter = (
        TremorFilter(cutoff_hz=tremor_cutoff_hz, sample_rate_hz=control_rate_hz, n_channels=3)
        if tremor_filter_enabled else None
    )

    print("[teleop] CoppeliaSim'e bağlanılıyor ve sahne kalibre ediliyor "
          "(eklemler q=0 olmalı)...")
    sim_client = CoppeliaSimClient()
    calibration = sim_client.calibrate_local_transforms()
    robot = CalibratedUR5Robot(calibration)
    q = sim_client.get_joint_positions()

    mouse_source = MouseInputSource()
    quit_signal = QuitOnEscape()

    print(f"[teleop] Kalibrasyon tamam. Ölçek: {scale_label} "
          f"(1 birim fare hareketi -> {SCALE_PRESETS[scale_label]:.2f} birim robot hareketi).")
    print(f"[teleop] Titreme filtresi: {'AÇIK' if tremor_filter_enabled else 'KAPALI'} "
          f"(cutoff={tremor_cutoff_hz} Hz).")
    print("[teleop] Fareyi hareket ettir: sağ/sol -> X, yukarı/aşağı -> Z. Çıkmak için ESC.")

    mouse_source.start()
    quit_signal.start()

    try:
        sim_client.start_simulation()
        while not quit_signal.should_quit:
            loop_start = time.perf_counter()

            master_delta_px = mouse_source.get_delta_and_reset()
            q = teleop_control_step(
                q, master_delta_px, pixel_to_mps, scaler, tremor_filter,
                dt, robot, damping=damping, joint_limits=joint_limits,
            )

            sim_client.set_joint_positions(q)
            if sim_client.stepping:
                sim_client.client.step()

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    finally:
        sim_client.stop_simulation()
        mouse_source.stop()
        quit_signal.stop()
        print("[teleop] Durduruldu.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoppeliaSim fare teleoperation")
    parser.add_argument("--scale", choices=list(SCALE_PRESETS.keys()), default="5:1")
    parser.add_argument("--no-tremor-filter", action="store_true",
                         help="Titreme filtresini kapat (ham fare girdisiyle karşılaştırmak için)")
    parser.add_argument("--tremor-cutoff", type=float, default=2.0, metavar="HZ")
    parser.add_argument("--rate", type=float, default=20.0, metavar="HZ",
                         help="Kontrol döngüsü frekansı")
    parser.add_argument("--sensitivity", type=float, default=0.002, metavar="M_PER_PIXEL",
                         help="1 piksel fare hareketinin kaç metre/saniye robot hızına karşılık geldiği")
    args = parser.parse_args()

    run_mouse_teleoperation(
        scale_label=args.scale,
        pixel_to_mps=args.sensitivity,
        tremor_filter_enabled=not args.no_tremor_filter,
        tremor_cutoff_hz=args.tremor_cutoff,
        control_rate_hz=args.rate,
    )


if __name__ == "__main__":
    main()
