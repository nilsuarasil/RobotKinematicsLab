import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.analysis.benchmark import run_solver_benchmark, format_benchmark_table, BenchmarkRow


def test_run_solver_benchmark_returns_one_row_per_solver():
    q_true = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    rows = run_solver_benchmark(target_position=target)
    assert len(rows) == 3
    names = {r.solver_name for r in rows}
    assert "Damped Least Squares" in names
    assert "Jacobian Pseudoinverse" in names
    assert "Jacobian Transpose" in names


def test_all_solvers_converge_on_easy_target():
    q_true = np.array([0.3, -0.4, 0.5, 0.2, -0.3, 0.1])
    target = forward_kinematics(q_true, UR5_DH_PARAMS)[:3, 3]

    rows = run_solver_benchmark(target_position=target, max_iters=1000, pos_tol=1e-3)
    for r in rows:
        assert r.converged, f"{r.solver_name} yakınsamadı"
        assert r.position_error_mm < 1.0
        assert r.elapsed_time_ms >= 0.0


def test_format_benchmark_table_contains_all_solver_names():
    rows = [
        BenchmarkRow("A", True, 10, 0.5, 0.0, 1.2),
        BenchmarkRow("B", True, 20, 0.8, 0.0, 2.3),
    ]
    table = format_benchmark_table(rows)
    assert "| A " in table or table.count("A") >= 1
    assert "0.50 mm" in table
    assert "1.2 ms" in table
    assert table.startswith("| Solver")
