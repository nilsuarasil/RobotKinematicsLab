"""
Singularity (tekillik) analizi: robotun bir konfigürasyonda hareket
serbestliği kaybettiği (bazı Cartesian yönlerde ne kadar hızlı hareket
ettirilirse ettirilsin sonsuz eklem hızı gerektiği) durumları tespit eder.

Üç ölçüt sunuyoruz:
  1. |det(J)| -- sadece KARE Jacobian için tanımlı (6-DOF, J tam 6x6 olduğunda
     geçerli). 0'a yaklaşması singularity işaretidir ama sayısal olarak
     kararsızdır (ölçek/birim'e duyarlı) -- SVD'ye tercih edilmez.
  2. Singular Value Decomposition (SVD): J = U Sigma V^T. En küçük tekil
     değer (sigma_min) 0'a yaklaşıyorsa, V^T'nin o singular vektörüne karşılık
     gelen Cartesian yönde robot hareket edemez hale gelir. Daha sağlam
     (birimlerden bağımsız, sayısal olarak kararlı) bir ölçüttür.
  3. Condition number: kappa(J) = sigma_max / sigma_min. Büyük kappa,
     Jacobian'ın "kötü şartlandırılmış" (ill-conditioned) olduğunu, yani
     küçük Cartesian hata/hız komutlarının çok büyük eklem hızlarına yol
     açabileceğini gösterir.
"""
from dataclasses import dataclass
import numpy as np


def singular_values(J):
    """J'nin tekil değerlerini büyükten küçüğe sıralı döndürür."""
    return np.linalg.svd(J, compute_uv=False)


def min_singular_value(J) -> float:
    return float(np.min(singular_values(J)))


def max_singular_value(J) -> float:
    return float(np.max(singular_values(J)))


def condition_number(J) -> float:
    """kappa(J) = sigma_max / sigma_min. sigma_min ~ 0 ise sonsuza yaklaşır
    (numpy.inf yerine çok büyük ama sonlu bir sayı döndürmek için sigma_min'e
    küçük bir epsilon ekleniyor)."""
    svals = singular_values(J)
    sigma_max, sigma_min = float(np.max(svals)), float(np.min(svals))
    return sigma_max / max(sigma_min, 1e-12)


def determinant_measure(J) -> float:
    """|det(J)| -- sadece J kare (n x n) olduğunda anlamlıdır; kare değilse
    None döner (örn. J_v tek başına 3x6 gibi kare olmayan durumlar için)."""
    if J.shape[0] != J.shape[1]:
        return None
    return float(abs(np.linalg.det(J)))


@dataclass
class SingularityReport:
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    determinant: "float | None"
    is_singular: bool
    status: str  # "SAFE" | "NEAR SINGULARITY" | "SINGULAR"


def analyze_singularity(J, sigma_threshold: float = 1e-3, condition_threshold: float = 1e4) -> SingularityReport:
    """
    J'nin singularity durumunu tam bir raporla döndürür.

    - sigma_min < sigma_threshold  -> "SINGULAR" (pratik olarak hareket
      serbestliği kaybedilmiş)
    - sigma_min < 10*sigma_threshold VEYA condition_number > condition_threshold
      -> "NEAR SINGULARITY" (dikkatli olunmalı, DLS gibi damped yöntemler
      tercih edilmeli)
    - aksi halde -> "SAFE"
    """
    svals = singular_values(J)
    sigma_min, sigma_max = float(np.min(svals)), float(np.max(svals))
    kappa = sigma_max / max(sigma_min, 1e-12)
    det = determinant_measure(J)

    if sigma_min < sigma_threshold:
        status = "SINGULAR"
    elif sigma_min < 10 * sigma_threshold or kappa > condition_threshold:
        status = "NEAR SINGULARITY"
    else:
        status = "SAFE"

    return SingularityReport(
        min_singular_value=sigma_min,
        max_singular_value=sigma_max,
        condition_number=kappa,
        determinant=det,
        is_singular=(status != "SAFE"),
        status=status,
    )
