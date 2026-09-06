"""
Üç IK solver'ını (Pseudoinverse, DLS, Jacobian Transpose) AYNI hedef üzerinde
çalıştırıp karşılaştırır -- iterasyon sayısı, pozisyon/orientation hatası ve
hesaplama süresi bakımından. Çıktı hem ham veri (BenchmarkRow listesi) hem de
doğrudan README/rapor için kullanılabilecek bir markdown tablosu olarak
üretilir.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Callable
import numpy as np

from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.ik_pseudoinverse import inverse_kinematics_pseudoinverse
from src.kinematics.ik_dls import inverse_kinematics_dls
from src.kinematics.ik_jacobian_transpose import inverse_kinematics_jacobian_transpose

DEFAULT_SOLVERS: Dict[str, Callable] = {
    "Jacobian Pseudoinverse": inverse_kinematics_pseudoinverse,
    "Damped Least Squares": inverse_kinematics_dls,
    "Jacobian Transpose": inverse_kinematics_jacobian_transpose,
}


@dataclass(eq=False)
class BenchmarkRow:
    solver_name: str
    converged: bool
    iterations: int
    position_error_mm: float
    orientation_error_deg: float
    elapsed_time_ms: float


def run_solver_benchmark(
    target_position,
    target_orientation=None,
    q_init=None,
    dh_params=UR5_DH_PARAMS,
    solvers: Optional[Dict[str, Callable]] = None,
    **solver_kwargs,
) -> List[BenchmarkRow]:
    """Her solver'ı aynı hedef + aynı başlangıç noktasından çalıştırır."""
    solvers = solvers if solvers is not None else DEFAULT_SOLVERS
    q_init = np.zeros(len(dh_params)) if q_init is None else np.asarray(q_init, dtype=float)

    rows = []
    for name, solver_fn in solvers.items():
        result = solver_fn(
            target_position=target_position, target_orientation=target_orientation,
            q_init=q_init.copy(), dh_params=dh_params, **solver_kwargs,
        )
        rows.append(BenchmarkRow(
            solver_name=name,
            converged=result.converged,
            iterations=result.iterations,
            position_error_mm=result.position_error * 1000.0,
            orientation_error_deg=float(np.degrees(result.orientation_error)),
            elapsed_time_ms=result.elapsed_time * 1000.0,
        ))
    return rows


def format_benchmark_table(rows: List[BenchmarkRow], include_orientation: bool = False) -> str:
    """
    README'de kullanılabilecek bir markdown tablosu üretir, örn.:

        | Solver | Iterations | Position Error | Time |
        |---|---:|---:|---:|
        | Jacobian Pseudoinverse | 28 | 0.72 mm | 4.1 ms |
    """
    if include_orientation:
        header = "| Solver | Iterations | Position Error | Orientation Error | Time |"
        sep = "|---|---:|---:|---:|---:|"
        lines = [header, sep]
        for r in rows:
            status = "" if r.converged else " (yakınsamadı)"
            lines.append(
                f"| {r.solver_name}{status} | {r.iterations} | {r.position_error_mm:.2f} mm | "
                f"{r.orientation_error_deg:.2f}° | {r.elapsed_time_ms:.1f} ms |"
            )
    else:
        header = "| Solver | Iterations | Position Error | Time |"
        sep = "|---|---:|---:|---:|"
        lines = [header, sep]
        for r in rows:
            status = "" if r.converged else " (yakınsamadı)"
            lines.append(
                f"| {r.solver_name}{status} | {r.iterations} | {r.position_error_mm:.2f} mm | "
                f"{r.elapsed_time_ms:.1f} ms |"
            )
    return "\n".join(lines)
