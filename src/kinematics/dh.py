"""
Denavit-Hartenberg (DH) parametreleri ve tek-eklem DH transform matrisi.

Standart DH konvansiyonu:
    T_i = Rz(theta_i) . Tz(d_i) . Tx(a_i) . Rx(alpha_i)

UR5 için standart DH tablosu (metre, radyan). Bu değerler literatürde yaygın
kullanılan UR5 DH parametreleridir; CoppeliaSim sahnendeki UR5 modelinin
gerçek link uzunluklarıyla küçük farklar olabilir -- entegrasyon aşamasında
Python FK'sı ile CoppeliaSim'in verdiği gerçek end-effector konumunu
karşılaştırarak bu farkı doğrula.
"""
import numpy as np

# (a, alpha, d) -- theta her eklemin değişkeni (q_i), burada sabit değil.
UR5_DH_PARAMS = [
    # a (m)      alpha (rad)   d (m)
    (0.0,        np.pi / 2,    0.089159),   # joint 1
    (-0.425,     0.0,          0.0),        # joint 2
    (-0.39225,   0.0,          0.0),        # joint 3
    (0.0,        np.pi / 2,    0.10915),    # joint 4
    (0.0,        -np.pi / 2,   0.09465),    # joint 5
    (0.0,        0.0,          0.0823),     # joint 6
]


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Tek bir DH satırından 4x4 homogeneous transform matrisi üretir."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])
