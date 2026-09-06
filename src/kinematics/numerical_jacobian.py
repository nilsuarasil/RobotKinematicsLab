"""
Analitik (geometric) Jacobian'ı SINAMAK için bağımsız bir sayısal (finite
difference) Jacobian hesaplayıcı. Amaç: jacobian.py'deki formülün (z_{i-1} x
(o_n - o_{i-1})) doğru türetildiğini, koda yazarken hata yapılmadığını
kod-bağımsız bir yöntemle doğrulamak -- "gerçekten matematiği uyguladığını
göstermek" (bkz. tests/test_numerical_jacobian.py).

Doğrusal (linear) kısım basit: merkezi fark.
    J_v[:, i] = (p(q + eps*e_i) - p(q - eps*e_i)) / (2*eps)

Açısal (angular) kısım biraz daha ince: rotasyon matrisleri vektör uzayında
değil (Lie grubu SO(3) üzerinde) yaşadığı için doğrudan çıkarma yapılamaz.
Küçük eps için, açısal hız ile rotasyon matrisi arasındaki ilişki:
    R(q+dq) ~= (I + eps*[w]_x) @ R(q)
yani:
    [w]_x ~= (R(q+eps) - R(q-eps)) @ R(q)^T / (2*eps)   (merkezi fark)
Burada [w]_x, w vektörünün "skew-symmetric" (çapraz-çarpım) matrisidir;
`vee()` bu matristen w vektörünü geri çıkarır (skew()'in tersi).
"""
import numpy as np


def skew(v):
    """3-vektör -> 3x3 skew-symmetric (çapraz çarpım) matris: skew(v) @ x == v x x."""
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def vee(S):
    """skew()'in tersi: 3x3 skew-symmetric matristen 3-vektörü çıkarır.
    Tam simetrik olmayabilir (sayısal hata) -- ortalama alınarak sağlamlaştırılır."""
    return np.array([
        0.5 * (S[2, 1] - S[1, 2]),
        0.5 * (S[0, 2] - S[2, 0]),
        0.5 * (S[1, 0] - S[0, 1]),
    ])


def numerical_geometric_jacobian(fk_func, q, eps: float = 1e-6):
    """
    fk_func(q) -> 4x4 homogeneous transform (world/base frame'e göre
    end-effector pose'u) döndüren herhangi bir forward-kinematics fonksiyonu.

    Döndürür: 6xN sayısal Jacobian (ilk 3 satır doğrusal, son 3 satır açısal).
    """
    q = np.asarray(q, dtype=float)
    n = len(q)
    J = np.zeros((6, n))

    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps

        T_plus = fk_func(q + dq)
        T_minus = fk_func(q - dq)

        p_plus, p_minus = T_plus[:3, 3], T_minus[:3, 3]
        J[:3, i] = (p_plus - p_minus) / (2 * eps)

        R_plus, R_minus = T_plus[:3, :3], T_minus[:3, :3]
        # R(q) merkez noktası olarak R_plus ve R_minus'un ortalamasını almak
        # yerine, w'nin tanımladığı bağıntıda R(q)^T yaklaşık olarak
        # R_minus^T (ya da R_plus^T) kullanılabilir; merkezi farkın simetrisini
        # korumak için R_plus @ R_minus^T kullanıyoruz (iki taraflı fark).
        dR = R_plus @ R_minus.T
        J[3:, i] = vee(dR) / (2 * eps)

    return J


def numerical_position_jacobian(fk_position_func, q, eps: float = 1e-6):
    """Sadece pozisyon döndüren fk fonksiyonları için 3xN sayısal Jacobian
    (V1'in position-only IK'sını doğrulamak için kullanılan basit versiyon)."""
    q = np.asarray(q, dtype=float)
    n = len(q)
    J = np.zeros((3, n))
    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps
        p_plus = np.asarray(fk_position_func(q + dq), dtype=float)
        p_minus = np.asarray(fk_position_func(q - dq), dtype=float)
        J[:, i] = (p_plus - p_minus) / (2 * eps)
    return J
