import os
import tempfile
from src.analysis.csv_logger import ExperimentLogger, FIELDS


def test_creates_file_with_header():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        ExperimentLogger(path)
        assert os.path.exists(path)
        with open(path) as f:
            header = f.readline().strip().split(",")
        assert header == FIELDS


def test_log_appends_rows():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        logger = ExperimentLogger(path)
        logger.log({"target_x": 0.1, "target_y": 0.2, "target_z": 0.3, "position_error_mm": 0.5})
        logger.log({"target_x": 0.4, "position_error_mm": 1.2})

        rows = logger.read_all()
        assert len(rows) == 2
        assert rows[0]["target_x"] == "0.1"
        assert rows[1]["position_error_mm"] == "1.2"


def test_reopening_existing_file_does_not_duplicate_header():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        logger1 = ExperimentLogger(path)
        logger1.log({"target_x": 1.0})

        logger2 = ExperimentLogger(path)  # aynı dosyayı yeniden aç
        logger2.log({"target_x": 2.0})

        with open(path) as f:
            lines = f.readlines()
        assert lines[0].startswith("timestamp,")
        assert len(lines) == 3  # header + 2 satır
