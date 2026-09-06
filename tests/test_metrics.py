from src.analysis.metrics import format_report


def test_report_marks_target_reached():
    report = format_report(
        target=[0.42, 0.18, 0.36],
        q_solution=[0.433, -0.545, 1.005, 0.0, 0.0, 0.0],
        actual_position=[0.4201, 0.1799, 0.3602],
        error_mm=1.7,
        reachable=True,
        converged=True,
    )
    assert "TARGET REACHED" in report
    assert "X = 0.420 m" in report
    assert "1.70 mm" in report


def test_report_marks_unreachable():
    report = format_report(
        target=[4.0, 0.0, 0.0],
        q_solution=[0, 0, 0, 0, 0, 0],
        actual_position=[0, 0, 0],
        error_mm=float("inf"),
        reachable=False,
        converged=False,
    )
    assert "UNREACHABLE TARGET" in report
    assert "IK Solution" not in report


def test_report_marks_not_reached_when_not_converged():
    report = format_report(
        target=[0.3, 0.1, 0.2],
        q_solution=[0, 0, 0, 0, 0, 0],
        actual_position=[0.9, 0.9, 0.9],
        error_mm=800.0,
        reachable=True,
        converged=False,
    )
    assert "TARGET NOT REACHED" in report
