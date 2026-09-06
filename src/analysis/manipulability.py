"""
Yoshikawa manipülabilite ölçüsü: w(q) = sqrt(det(J J^T)).

Geometrik yorumu: eklem hızları birim küre içinde (||q_dot|| <= 1) iken,
ulaşılabilecek Cartesian hızların oluşturduğu elipsoidin HACMİYLE orantılıdır.
w büyükse robot o konfigürasyonda her yöne rahatça hareket edebilir; w -> 0
ise singularity'ye yaklaşılıyor demektir (elipsoid bir veya daha fazla
eksende çöküyor).

Not: jacobian.py'de de basit bir `manipulability()` fonksiyonu var (geriye
dönük uyumluluk için orada bırakıldı); burada aynı formülün analiz katmanına
ait, SVD ile de çapraz doğrulanmış (w = sigma_1 * sigma_2 * ... * sigma_m)
versiyonu bulunuyor.
"""
import numpy as np


def yoshikawa_manipulability(J) -> float:
    """w(q) = sqrt(det(J J^T)). J kare değilse de (m x n, m<n) çalışır."""
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(det, 0.0)))


def manipulability_from_singular_values(J) -> float:
    """Çapraz doğrulama: w = sigma_1 * sigma_2 * ... * sigma_m (tüm tekil
    değerlerin çarpımı) -- yoshikawa_manipulability ile matematiksel olarak
    özdeş olmalı (bkz. tests/test_manipulability.py)."""
    svals = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(svals))


def manipulability_ellipsoid_axes(J):
    """
    Manipülabilite elipsoidinin yarı-eksen uzunlukları: J J^T'nin özdeğerlerinin
    kareköküdür (= J'nin tekil değerleri). Elipsoidin yönleri J J^T'nin
    özvektörleridir (U matrisi, SVD'den).
    """
    U, svals, _ = np.linalg.svd(J)
    return svals, U
