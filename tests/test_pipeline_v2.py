"""
main.py'nin V2 dry-run CLI yollarını (orientation hedefi, solver seçimi,
multi-start, benchmark, CSV export) uçtan uca test eder. V1 davranışının
(hiçbir V2 bayrağı verilmediğinde) DEĞİŞMEDİĞİni de doğrular -- bu kritik,
çünkü --sim modu gerçek donanımda doğrulanmış V1 koduna dayanıyor.
"""
import os
import tempfile
import numpy as np
from src.main import run, run_tremor_demo


def test_v1_dry_run_unchanged_without_v2_flags():
    report = run(target=[0.42, 0.18, 0.36], verbose=False)
    assert "TARGET REACHED" in report
    assert "IK Solution:" in report
    assert "Final Position Error:" in report


def test_v2_orientation_target_produces_orientation_section():
    report = run(target=[0.3, 0.2, 0.4], orientation=[30.0, 10.0, -20.0], verbose=False)
    assert "Roll  = 30.0°" in report
    assert "Final Orientation Error:" in report
    assert "Manipulability Analysis:" in report


def test_v2_solver_selection_all_three():
    for solver in ("pseudoinverse", "dls", "transpose"):
        report = run(target=[0.35, 0.1, 0.3], solver=solver, verbose=False)
        assert "TARGET REACHED" in report


def test_v2_multi_start_reports_selected_candidate():
    report = run(target=[0.4, -0.1, 0.35], multi_start=True, verbose=False)
    assert "multi-start" in report
    assert "TARGET REACHED" in report


def test_v2_benchmark_returns_table_not_report():
    result = run(target=[0.42, 0.18, 0.36], benchmark=True, verbose=False)
    assert result.startswith("| Solver")
    assert "Jacobian Pseudoinverse" in result
    assert "Damped Least Squares" in result
    assert "Jacobian Transpose" in result


def test_v2_unreachable_target_reports_correctly():
    report = run(target=[5.0, 5.0, 5.0], solver="dls", verbose=False)
    assert "UNREACHABLE TARGET" in report


def test_v2_csv_export_writes_row():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "log.csv")
        run(target=[0.42, 0.18, 0.36], solver="dls", csv_path=csv_path, verbose=False)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 sonuç satırı


def test_tremor_demo_reports_reduction_percentage():
    text = run_tremor_demo(verbose=False)
    assert "Tremor Band Reduction" in text
    assert "%" in text


def test_v1_csv_export_also_works_without_v2_flags():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "log.csv")
        run(target=[0.42, 0.18, 0.36], csv_path=csv_path, verbose=False)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
