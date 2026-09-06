"""
Kullanıcı tarafından tasarlanan proje raporu formatını üretir:

Target:
X = 0.42 m
Y = 0.18 m
Z = 0.36 m

IK Solution:
q1 = 24.8°
...

Final Position Error:
1.7 mm

Status:
TARGET REACHED
"""
import numpy as np


def format_report(target, q_solution, actual_position, error_mm: float,
                   reachable: bool, converged: bool,
                   error_threshold_mm: float = 5.0) -> str:
    target = np.asarray(target, dtype=float)
    q_deg = np.degrees(np.asarray(q_solution, dtype=float))

    lines = ["Target:",
             f"X = {target[0]:.3f} m",
             f"Y = {target[1]:.3f} m",
             f"Z = {target[2]:.3f} m",
             ""]

    if reachable:
        lines.append("IK Solution:")
        for i, angle in enumerate(q_deg, start=1):
            lines.append(f"q{i} = {angle:.1f}°")
        lines.append("")
        lines.append("Final Position Error:")
        lines.append(f"{error_mm:.2f} mm")
        lines.append("")

    lines.append("Status:")
    if not reachable:
        lines.append("UNREACHABLE TARGET")
    elif converged and error_mm <= error_threshold_mm:
        lines.append("TARGET REACHED")
    else:
        lines.append("TARGET NOT REACHED (IK did not converge within tolerance)")

    return "\n".join(lines)


def format_report_v2(
    target_position, q_solution, actual_position,
    position_error_mm: float, reachable: bool, converged: bool,
    solver_name: str = "", iterations: int = 0,
    target_orientation_deg=None, orientation_error_deg: float = 0.0,
    condition_number: float = None, manipulability: float = None,
    singularity_status: str = None,
    error_threshold_mm: float = 5.0,
) -> str:
    """
    V2 raporu: V1'in Target/IK Solution/Final Position Error/Status formatını
    korur, üzerine solver bilgisi, orientation hatası (hedeflendiyse) ve
    singularity/manipulability analizini ekler (bkz. V2 arayüzü mockup'ı).
    """
    target_position = np.asarray(target_position, dtype=float)
    q_deg = np.degrees(np.asarray(q_solution, dtype=float))

    lines = ["Target:",
             f"X = {target_position[0]:.3f} m",
             f"Y = {target_position[1]:.3f} m",
             f"Z = {target_position[2]:.3f} m"]
    if target_orientation_deg is not None:
        roll, pitch, yaw = target_orientation_deg
        lines += [f"Roll  = {roll:.1f}°", f"Pitch = {pitch:.1f}°", f"Yaw   = {yaw:.1f}°"]
    lines.append("")

    if reachable:
        lines.append(f"IK Solver: {solver_name} ({iterations} iterations)")
        lines.append("IK Solution:")
        for i, angle in enumerate(q_deg, start=1):
            lines.append(f"q{i} = {angle:.1f}°")
        lines.append("")
        lines.append("Final Position Error:")
        lines.append(f"{position_error_mm:.2f} mm")
        if target_orientation_deg is not None:
            lines.append("Final Orientation Error:")
            lines.append(f"{orientation_error_deg:.2f}°")
        lines.append("")

        if condition_number is not None or manipulability is not None:
            lines.append("Manipulability Analysis:")
            if manipulability is not None:
                lines.append(f"Manipulability     = {manipulability:.4f}")
            if condition_number is not None:
                lines.append(f"Condition Number   = {condition_number:.1f}")
            if singularity_status is not None:
                marker = "⚠ " if singularity_status != "SAFE" else ""
                lines.append(f"Singularity Status = {marker}{singularity_status}")
            lines.append("")

    lines.append("Status:")
    if not reachable:
        lines.append("UNREACHABLE TARGET")
    elif converged and position_error_mm <= error_threshold_mm:
        lines.append("TARGET REACHED")
    else:
        lines.append("TARGET NOT REACHED (IK did not converge within tolerance)")

    return "\n".join(lines)
