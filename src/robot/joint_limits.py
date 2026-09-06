"""
UR5 için varsayılan eklem limitleri (radyan).

NOT: Gerçek UR5 donanımı yazılımsal olarak her eklemde tipik olarak ±360°
(±2π) izin verir. Burada, IK çözümlerinin makul ve çarpışma riski daha düşük
konfigürasyonlarda kalmasını sağlamak için daha muhafazakar, ±180° (±π)
varsayılan limitler kullanıyoruz. CoppeliaSim sahnendeki UR5 modelinin gerçek
limitleriyle eşleştirmek istersen bu listeyi güncelle.
"""
import numpy as np

JOINT_LIMITS = [
    (-np.pi, np.pi),  # J1
    (-np.pi, np.pi),  # J2
    (-np.pi, np.pi),  # J3
    (-np.pi, np.pi),  # J4
    (-np.pi, np.pi),  # J5
    (-np.pi, np.pi),  # J6
]


def check_joint_limits(q, limits=JOINT_LIMITS):
    """
    q'daki her eklem açısının limit içinde olup olmadığını kontrol eder.
    Döndürür: limit dışı kalan eklemlerin (index, açı, limit) listesi
    -- boşsa hepsi limit içinde demektir.
    """
    violations = []
    for i, (angle, (lo, hi)) in enumerate(zip(q, limits)):
        if angle < lo or angle > hi:
            violations.append((i, angle, (lo, hi)))
    return violations


def clamp_to_limits(q, limits=JOINT_LIMITS):
    """Her eklemi kendi limit aralığına sıkıştırır (clip)."""
    return [float(np.clip(angle, lo, hi)) for angle, (lo, hi) in zip(q, limits)]
