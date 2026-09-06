"""
Cartesian Trajectory: V1'in "eklem-uzayında" (joint-space) interpolasyonunun
aksine, end-effector'ün UÇ NOKTASININ (Cartesian uzayda) DÜZ BİR ÇİZGİ
izlemesini sağlar:

    p(t) = p0 + s(t) * (pf - p0)

Burada s(t), quintic.py'deki zaman ölçekleme fonksiyonu (hız/ivme uçlarda 0).
Bu, eklem-uzayı interpolasyonundan farklı bir sonuç verir: eklem-uzayında
düz giden bir hareket, end-effector'ün Cartesian uzayda EĞRİ bir yol
izlemesine sebep olabilir (özellikle büyük hareketlerde) -- cerrahi
robotikte "aletin düz bir çizgide ilerlemesi" çoğunlukla kritik bir
gereksinimdir, bu yüzden Cartesian trajectory tercih edilir.

Her Cartesian noktasına ulaşmak için IK'nın (ik_dls.py gibi) her adımda
çağrılması gerekir -- bkz. main.py'deki V2 pipeline entegrasyonu.
"""
import numpy as np
from src.planning.quintic import quintic_scaling_profile


def cartesian_trajectory(p_start, p_end, steps: int = 50):
    """(steps, 3) şeklinde, quintic zaman ölçekli düz-çizgi Cartesian trajectory."""
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    s = quintic_scaling_profile(steps)
    return np.array([p_start + si * (p_end - p_start) for si in s])
