"""
Robot Kinematics Lab -- ana pipeline.

V1 (varsayılan, hiçbir V2 bayrağı verilmezse davranış birebir aynı kalır):
    python -m src.main --target 0.42 0.18 0.36
    python -m src.main --target 0.42 0.18 0.36 --sim

V2 (--orientation / --solver / --multi-start / --benchmark bayraklarından
herhangi biri verilirse devreye girer -- şu an sadece DRY-RUN modunda,
yani --sim OLMADAN kullanılabilir; --sim hâlâ V1'in kanıtlanmış, gerçek
donanımda 0.5mm hata veren kalibrasyonlu pozisyon-only yolunu kullanır):

    # 6D pose hedefi (pozisyon + yönelim, roll/pitch/yaw derece cinsinden)
    python -m src.main --target 0.42 0.18 0.36 --orientation 90 15 -30

    # Solver seçimi (pseudoinverse | dls | transpose)
    python -m src.main --target 0.42 0.18 0.36 --solver transpose

    # Birden fazla başlangıç noktasından çözüp en iyi konfigürasyonu seç
    python -m src.main --target 0.42 0.18 0.36 --multi-start

    # Üç solver'ı karşılaştıran bir tablo yazdır (robotu hareket ettirmez)
    python -m src.main --target 0.42 0.18 0.36 --benchmark

    # Sonucu bir CSV dosyasına da kaydet
    python -m src.main --target 0.42 0.18 0.36 --csv results/log.csv

--sim verilmezse "dry-run" modunda çalışır: CoppeliaSim'e bağlanmadan, IK
çözümünün kendi Forward Kinematics'imizle tutarlılığını raporlar.
--sim verilirse gerçek CoppeliaSim'e bağlanır (bkz. simulation/coppelia_client.py).
"""
import argparse
import numpy as np

from src.robot.robot_model import UR5Robot, CalibratedUR5Robot
from src.planning.trajectory import quintic_joint_trajectory
from src.analysis.pose_error import position_error_mm, orientation_error_deg
from src.analysis.metrics import format_report, format_report_v2
from src.kinematics.jacobian import compute_jacobian
from src.analysis.singularity import analyze_singularity
from src.analysis.manipulability import yoshikawa_manipulability
from src.kinematics._ik_common import resolve_target_rotation
from src.kinematics.transforms import inverse_transform, transform_point


def _run_v2_dry_run(robot, target, orientation, solver, multi_start, benchmark,
                     q_current, csv_path, verbose):
    """
    V2 araştırma modu: custom solver seçimi, 6D pose hedefi, çoklu-başlangıç
    IK ve solver karşılaştırma tablosu. Sadece dry-run'da (CoppeliaSim olmadan)
    çalışır -- gerçek robotu hareket ettirmez, sadece Python matematiğini
    çalıştırır/raporlar.
    """
    solver_key = solver or "dls"

    reachable, distance = robot.is_reachable(target)
    if not reachable:
        report = format_report_v2(
            target_position=target, q_solution=q_current, actual_position=q_current,
            position_error_mm=float("inf"), reachable=False, converged=False,
        )
        if verbose:
            print(f"Hedef robot tabanından {distance:.3f} m uzakta (max erişim aşıldı ya da çok yakın).")
            print(report)
        return report

    target_orientation_rad = None
    target_orientation_deg = None
    if orientation is not None:
        target_orientation_deg = orientation
        target_orientation_rad = np.radians(orientation)

    if benchmark:
        from src.analysis.benchmark import run_solver_benchmark, format_benchmark_table

        base_inv = inverse_transform(robot.base_transform)
        local_target = transform_point(base_inv, target)
        local_R = None
        if target_orientation_rad is not None:
            R_world = resolve_target_rotation(target_orientation_rad)
            local_R = base_inv[:3, :3] @ R_world

        rows = run_solver_benchmark(
            target_position=local_target, target_orientation=local_R,
            q_init=q_current, dh_params=robot.dh_params,
        )
        table = format_benchmark_table(rows, include_orientation=(local_R is not None))
        if verbose:
            print("Solver Comparison:")
            print(table)
        return table

    if multi_start:
        multi_result = robot.solve_ik_v2(
            target, target_orientation=target_orientation_rad, q_init=q_current,
            solver=solver_key, use_multi_start=True,
        )
        selected = multi_result.selected
        q_solution = selected.q
        converged = selected.converged
        solver_label = f"{solver_key} (multi-start, seçilen: {selected.label}, {len(multi_result.candidates)} aday)"
        iterations = 0
        if verbose:
            print(f"[multi-start] {len(multi_result.candidates)} aday değerlendirildi, seçilen: {selected.label} (cost={selected.cost:.3f})")
    else:
        result = robot.solve_ik_v2(
            target, target_orientation=target_orientation_rad, q_init=q_current, solver=solver_key,
        )
        q_solution = result.q
        converged = result.converged
        solver_label = result.solver_name
        iterations = result.iterations

        if verbose and result.joint_limit_violations:
            print("UYARI: IK çözümü joint limitlerini aşıyor:")
            for idx, angle, (lo, hi) in result.joint_limit_violations:
                print(f"  J{idx + 1} = {np.degrees(angle):.1f}° "
                      f"(limit: [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°])")

    T_achieved_world = robot.forward(q_solution)
    actual_position = T_achieved_world[:3, 3]
    err_mm = position_error_mm(actual_position, target)

    ori_err_deg = 0.0
    if target_orientation_rad is not None:
        target_R_world = resolve_target_rotation(target_orientation_rad)
        ori_err_deg = orientation_error_deg(T_achieved_world[:3, :3], target_R_world)

    J = compute_jacobian(q_solution, robot.dh_params)
    singularity_report = analyze_singularity(J)
    manip = yoshikawa_manipulability(J)

    report = format_report_v2(
        target_position=target, q_solution=q_solution, actual_position=actual_position,
        position_error_mm=err_mm, reachable=True, converged=converged,
        solver_name=solver_label, iterations=iterations,
        target_orientation_deg=target_orientation_deg, orientation_error_deg=ori_err_deg,
        condition_number=singularity_report.condition_number, manipulability=manip,
        singularity_status=singularity_report.status,
    )
    if verbose:
        print(report)

    if csv_path:
        from src.analysis.csv_logger import ExperimentLogger
        logger = ExperimentLogger(csv_path)
        record = {
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
            "actual_x": actual_position[0], "actual_y": actual_position[1], "actual_z": actual_position[2],
            "position_error_mm": err_mm, "orientation_error_deg": ori_err_deg,
            "condition_number": singularity_report.condition_number, "manipulability": manip,
            "solver_name": solver_label, "solver_iterations": iterations,
            "converged": converged,
        }
        for i, angle in enumerate(q_solution, start=1):
            record[f"q{i}"] = float(angle)
        logger.log(record)
        if verbose:
            print(f"[csv] Kayıt eklendi: {csv_path}")

    return report


def run(target, use_sim: bool = False, q_current=None, steps: int = 50, verbose: bool = True,
        orientation=None, solver: str = None, multi_start: bool = False,
        benchmark: bool = False, csv_path: str = None) -> str:
    use_v2 = any([orientation is not None, solver is not None, multi_start, benchmark])

    sim_client = None

    if use_sim:
        if use_v2 and verbose:
            print("[uyarı] --sim modu şu an sadece V1 (pozisyon-only, kalibrasyonlu) "
                  "IK kullanıyor; --orientation/--solver/--multi-start/--benchmark "
                  "sadece dry-run modunda (--sim OLMADAN) etkilidir.")

        # CoppeliaSim'e IK hesabından ÖNCE bağlanıyoruz. DH tablosuna
        # (literatür değerlerine) güvenmek yerine, robotun bu spesifik
        # sahnedeki GERÇEK geometrisini (taban pose'u + her eklemin bir
        # öncekine göre gerçek montaj transformu) doğrudan CoppeliaSim'den
        # okuyup CalibratedUR5Robot ile kullanıyoruz -- bu, DH tablosu ile
        # CoppeliaSim'in bu UR5 modelinin "sıfır açı" referansı arasında
        # oluşabilecek uyuşmazlığı (ki test sırasında karşılaştık) kökten
        # ortadan kaldırıyor. ÖNKOŞUL: eklemler henüz hareket etmemiş (q=0)
        # olmalı -- bu yüzden bağlantıdan hemen sonra, hiçbir şey
        # hareket ettirmeden kalibrasyonu yapıyoruz.
        from src.simulation.coppelia_client import CoppeliaSimClient
        sim_client = CoppeliaSimClient()
        calibration = sim_client.calibrate_local_transforms()
        robot = CalibratedUR5Robot(calibration)
        if verbose:
            print(f"[sim] Robot taban konumu (dünya): {robot.base_position}")
            print(f"[sim] Sahneden {robot.n_joints} eklemin gerçek geometrisi kalibre edildi.")
    else:
        # Dry-run: CoppeliaSim yok, literatür DH tablosunu kullanan modeli kullan.
        robot = UR5Robot()

    q_current = np.array(q_current, dtype=float) if q_current is not None else np.zeros(robot.n_joints)

    if use_v2 and not use_sim:
        return _run_v2_dry_run(robot, target, orientation, solver, multi_start, benchmark,
                                q_current, csv_path, verbose)

    reachable, distance = robot.is_reachable(target)
    if not reachable:
        report = format_report(
            target=target, q_solution=q_current,
            actual_position=robot.end_effector_position(q_current),
            error_mm=float("inf"), reachable=False, converged=False,
        )
        if verbose:
            print(f"Hedef robot tabanından {distance:.3f} m uzakta (max erişim aşıldı ya da çok yakın).")
            print(report)
        return report

    ik_result = robot.solve_ik(target, q_init=q_current)

    if verbose and ik_result.joint_limit_violations:
        print("UYARI: IK çözümü joint limitlerini aşıyor:")
        for idx, angle, (lo, hi) in ik_result.joint_limit_violations:
            print(f"  J{idx + 1} = {np.degrees(angle):.1f}° "
                  f"(limit: [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°])")

    trajectory = quintic_joint_trajectory(q_current, ik_result.q, steps=steps)

    if use_sim:
        if verbose:
            print(f"[sim] Bağlanılan joint sayısı: {len(sim_client.joint_handles)}")
            print(f"[sim] Hareketten önce joint açıları (deg): "
                  f"{np.degrees(sim_client.get_joint_positions())}")

        # try/finally: start_simulation() ile açılan "stepping" modunun, bir
        # hata durumunda bile mutlaka stop_simulation() ile kapatılmasını
        # garantiler -- aksi halde CoppeliaSim dışarıdan adım bekler halde
        # kilitli kalabilir (bkz. coppelia_client.py'deki not).
        try:
            sim_client.start_simulation()
            sim_client.send_trajectory(trajectory)
            actual_position = sim_client.get_end_effector_position()

            if verbose:
                print(f"[sim] Hareketten sonra joint açıları (deg): "
                      f"{np.degrees(sim_client.get_joint_positions())}")
                print(f"[sim] Hedef joint açıları (deg):           "
                      f"{np.degrees(ik_result.q)}")
        finally:
            sim_client.stop_simulation()
    else:
        # Dry-run: CoppeliaSim yok, kendi FK'mızla "gerçekleşen" konumu hesaplıyoruz.
        actual_position = robot.end_effector_position(ik_result.q)

    error_mm = position_error_mm(actual_position, target)

    report = format_report(
        target=target,
        q_solution=ik_result.q,
        actual_position=actual_position,
        error_mm=error_mm,
        reachable=True,
        converged=ik_result.converged,
    )
    if verbose:
        print(report)

    if csv_path:
        from src.analysis.csv_logger import ExperimentLogger
        logger = ExperimentLogger(csv_path)
        record = {
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
            "actual_x": actual_position[0], "actual_y": actual_position[1], "actual_z": actual_position[2],
            "position_error_mm": error_mm, "solver_name": "V1 (position-only)",
            "converged": ik_result.converged,
        }
        for i, angle in enumerate(ik_result.q, start=1):
            record[f"q{i}"] = float(angle)
        logger.log(record)
        if verbose:
            print(f"[csv] Kayıt eklendi: {csv_path}")

    return report


def run_tremor_demo(cutoff_hz: float = 2.0, sample_rate_hz: float = 100.0, verbose: bool = True) -> str:
    """
    CoppeliaSim/robot gerektirmeyen bağımsız bir demo: sentetik bir "cerrah
    eli" sinyali (yavaş kasıtlı hareket + yüksek frekanslı titreme) üretir,
    TremorFilter ile filtreler ve titreme bandındaki enerjinin ne kadar
    azaldığını raporlar. `python -m src.main --tremor-demo` ile çalıştırılır.
    """
    from src.control.tremor_filter import (
        TremorFilter, generate_synthetic_hand_signal,
        analyze_tremor_reduction, format_tremor_demo_report,
    )

    raw = generate_synthetic_hand_signal(sample_rate_hz=sample_rate_hz, n_channels=3)
    tf = TremorFilter(cutoff_hz=cutoff_hz, sample_rate_hz=sample_rate_hz, n_channels=3)
    filtered = tf.filter_signal(raw)

    report = analyze_tremor_reduction(raw, filtered, sample_rate_hz=sample_rate_hz)
    text = format_tremor_demo_report(report, cutoff_hz, sample_rate_hz)
    if verbose:
        print(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="Robot Kinematics Lab")
    parser.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--tremor-demo", action="store_true",
                         help="CoppeliaSim gerektirmeyen titreme-filtresi demosunu çalıştır (--target gerekmez)")
    parser.add_argument("--sim", action="store_true",
                         help="CoppeliaSim'e gerçekten bağlan (varsayılan: dry-run, sadece Python FK/IK)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--orientation", type=float, nargs=3, default=None, metavar=("ROLL", "PITCH", "YAW"),
                         help="Hedef yönelim, derece cinsinden roll/pitch/yaw (V2, sadece dry-run)")
    parser.add_argument("--solver", choices=["pseudoinverse", "dls", "transpose", "joint_limit_avoidance"],
                         default=None,
                         help="V2 IK solver seçimi (V2, sadece dry-run; belirtilmezse V1 davranışı korunur). "
                              "'joint_limit_avoidance': DLS + null-space projection, eklemleri limit "
                              "merkezine yakın tutmaya çalışır.")
    parser.add_argument("--multi-start", action="store_true",
                         help="Birden fazla başlangıç noktasından çözüp en iyi konfigürasyonu seç (V2)")
    parser.add_argument("--benchmark", action="store_true",
                         help="Üç solver'ı karşılaştıran bir tablo yazdır, robotu hareket ettirme (V2)")
    parser.add_argument("--csv", type=str, default=None, metavar="PATH",
                         help="Sonucu bu CSV dosyasına da kaydet (bkz. analysis/csv_logger.py)")
    args = parser.parse_args()

    if args.tremor_demo:
        run_tremor_demo()
        return

    if args.target is None:
        parser.error("--target gerekli (--tremor-demo kullanmıyorsan)")

    run(target=args.target, use_sim=args.sim, steps=args.steps,
        orientation=args.orientation, solver=args.solver,
        multi_start=args.multi_start, benchmark=args.benchmark, csv_path=args.csv)


if __name__ == "__main__":
    main()
