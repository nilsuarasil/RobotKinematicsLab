"""
Motion Scaling: cerrahi robotik açısından kritik bir kavram -- cerrahın elinin
(master input) hareketi, robota (slave/tool tip) DAHA KÜÇÜK bir hareket olarak
uygulanır. Bu hem hassasiyeti artırır (büyük el hareketi -> küçük, titrek
olmayan alet hareketi) hem de titremeyi (tremor) doğal olarak biraz bastırır
(3. proje "Surgical Tremor Filter" bu katmanın üzerine inşa edilecek).

    robot_motion = master_motion * scale_factor

Ölçek genelde "N:1" olarak ifade edilir (N birim master hareketi -> 1 birim
robot hareketi), yani scale_factor = 1/N.
"""
import numpy as np

# "N:1" etiketi -> scale_factor = 1/N (örn. "5:1" -> master 5 birim
# hareket ederse robot 1 birim hareket eder, yani carpim faktoru 0.2).
SCALE_PRESETS = {
    "1:1": 1.0,
    "2:1": 0.5,
    "5:1": 0.2,
    "10:1": 0.1,
}


def scale_motion(input_delta, scale_factor: float):
    """master_delta * scale_factor -- input_delta skaler ya da vektör olabilir."""
    return np.asarray(input_delta, dtype=float) * scale_factor


class MotionScaler:
    """Bir teleoperation oturumu boyunca ölçek faktörünü tutan basit durum
    nesnesi -- UI'da "MOTION SCALE: 1:1 / 2:1 / 5:1 / 10:1" seçimine karşılık gelir."""

    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    @classmethod
    def from_preset(cls, label: str) -> "MotionScaler":
        if label not in SCALE_PRESETS:
            raise ValueError(f"Bilinmeyen ölçek etiketi: {label!r}. Seçenekler: {list(SCALE_PRESETS)}")
        return cls(SCALE_PRESETS[label])

    def set_scale(self, scale: float):
        self.scale = float(scale)

    def set_preset(self, label: str):
        self.scale = SCALE_PRESETS[label]

    def apply(self, master_delta):
        return scale_motion(master_delta, self.scale)
