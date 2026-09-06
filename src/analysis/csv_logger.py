"""
CSV Export: her IK/hareket denemesinin tam kaydını (hedef, gerçekleşen konum,
hatalar, eklem açıları, condition number, solver süresi...) bir CSV dosyasına
ekler -- deney sonuçlarını daha sonra Excel/pandas ile analiz etmek için.
"""
import csv
import os
import time
from typing import Optional


FIELDS = [
    "timestamp",
    "target_x", "target_y", "target_z",
    "actual_x", "actual_y", "actual_z",
    "position_error_mm", "orientation_error_deg",
    "q1", "q2", "q3", "q4", "q5", "q6",
    "condition_number", "manipulability",
    "solver_name", "solver_iterations", "solver_time_ms", "converged",
]


class ExperimentLogger:
    """Her `log()` çağrısında CSV dosyasına bir satır ekler. Dosya yoksa
    başlık satırıyla birlikte oluşturur; varsa sonuna ekler (append)."""

    def __init__(self, csv_path: str, fields=FIELDS):
        self.csv_path = csv_path
        self.fields = fields
        self._ensure_header()

    def _ensure_header(self):
        file_exists = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        if not file_exists:
            directory = os.path.dirname(self.csv_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()

    def log(self, record: dict, timestamp: Optional[float] = None):
        """record: FIELDS'ın bir alt kümesini içeren dict (eksik alanlar boş
        bırakılır). timestamp verilmezse otomatik olarak `time.time()` kullanılır."""
        row = {field: "" for field in self.fields}
        row["timestamp"] = timestamp if timestamp is not None else time.time()
        row.update({k: v for k, v in record.items() if k in row})

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(row)

    def read_all(self):
        """Test/doğrulama amaçlı: yazılmış tüm satırları dict listesi olarak okur."""
        with open(self.csv_path, "r", newline="") as f:
            return list(csv.DictReader(f))
