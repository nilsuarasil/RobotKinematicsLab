"""V2'nin 6D-pose-aware IK çözücülerinin (pseudoinverse/DLS/transpose) ortak
sonuç tipi. V1'in position-only IKResult'ından (inverse.py) ayrı tutuluyor ki
V1 kodu/testleri bozulmasın -- burada ek olarak orientation hatası ve
karşılaştırma/benchmark için solver adı + geçen süre de tutulur."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass(eq=False)
class PoseIKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    position_error: float          # metre
    orientation_error: float       # radyan (orientation hedeflenmediyse 0.0)
    joint_limit_violations: List = field(default_factory=list)
    solver_name: str = ""
    elapsed_time: float = 0.0      # saniye (benchmark.py için)
