"""
Trajectory Planning: q_start'tan q_target'a robotun "ışınlanmadan",
kademeli olarak hareket etmesini sağlayacak ara noktalar (waypoint) üretir.

- Linear: her eklem açısı sabit hızla değişir. Basit ama süreksiz ivmeye
  sahiptir (başlangıç/bitişte ani hız değişimi -- gerçek donanımda sarsıntı
  yaratabilir).
- Quintic (5. derece polinom): hız VE ivme başlangıç/bitişte sıfır olacak
  şekilde yumuşak zaman ölçekleme (s-curve) uygular. Gerçek robotlarda
  tercih edilen yöntemdir.
"""
import numpy as np


def linear_joint_trajectory(q_start, q_target, steps: int = 50):
    """(steps, n_joints) şeklinde, sabit hızlı doğrusal eklem-uzayı trajectory'si."""
    q_start = np.asarray(q_start, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    t = np.linspace(0.0, 1.0, steps)
    return np.array([q_start + (q_target - q_start) * ti for ti in t])


def quintic_joint_trajectory(q_start, q_target, steps: int = 50):
    """
    s(t) = 10t^3 - 15t^4 + 6t^5 zaman ölçekleme fonksiyonuyla quintic
    trajectory. t normalize [0,1]; s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0.
    """
    q_start = np.asarray(q_start, dtype=float)
    q_target = np.asarray(q_target, dtype=float)
    t = np.linspace(0.0, 1.0, steps)
    s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    return np.array([q_start + (q_target - q_start) * si for si in s])
