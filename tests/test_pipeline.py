from src.main import run


def test_pipeline_reaches_reachable_target_dry_run():
    report = run(target=[0.3, 0.2, 0.3], use_sim=False, verbose=False)
    assert "TARGET REACHED" in report
    assert "Final Position Error" in report


def test_pipeline_reports_unreachable_target():
    report = run(target=[4.0, 0.0, 0.0], use_sim=False, verbose=False)
    assert "UNREACHABLE TARGET" in report
