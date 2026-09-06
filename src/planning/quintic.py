"""
Quintic (5. derece polinom) zaman ölçekleme: s(t), t in [0,1] normalize
zamanını, "ilerleme yüzdesi" s in [0,1]'e çevirir öyle ki:

    s(0) = 0,  s(1) = 1
    s'(0) = 0, s'(1) = 0      (hız sıfır -- ani başlangıç/duruş yok)
    s''(0) = 0, s''(1) = 0    (ivme sıfır -- sarsıntısız/"jerk-free" uçlar)

    s(t)  = 10t^3 - 15t^4 + 6t^5

Bu tek boyutlu s(t) fonksiyonu hem eklem-uzayı (joint_trajectory.py/trajectory.py)
hem de Cartesian-uzayı (cartesian_trajectory.py, slerp.py) trajectory'lerinde
AYNI zaman profilini kullanmak için ortak bir yerde tutuluyor.
"""
import numpy as np


def quintic_time_scaling(t):
    """s(t) = 10t^3 - 15t^4 + 6t^5. t: skaler ya da array, [0,1] aralığında."""
    t = np.asarray(t, dtype=float)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def quintic_time_scaling_derivative(t):
    """ds/dt = 30t^2 - 60t^3 + 30t^4 -- "hız" profili (t=0 ve t=1'de 0)."""
    t = np.asarray(t, dtype=float)
    return 30 * t ** 2 - 60 * t ** 3 + 30 * t ** 4


def quintic_time_scaling_second_derivative(t):
    """d^2s/dt^2 = 60t - 180t^2 + 120t^3 -- "ivme" profili (t=0 ve t=1'de 0)."""
    t = np.asarray(t, dtype=float)
    return 60 * t - 180 * t ** 2 + 120 * t ** 3


def quintic_scaling_profile(steps: int):
    """[0,1] aralığında `steps` adet zaman noktası için s(t) dizisini döndürür."""
    t = np.linspace(0.0, 1.0, steps)
    return quintic_time_scaling(t)
