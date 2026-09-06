"""
Orientation (yönelim) matematiği: rotasyon matrisi <-> Euler (Roll-Pitch-Yaw)
<-> quaternion dönüşümleri, quaternion cebiri, SLERP ve 6D pose IK için
gereken "orientation error" (açısal hata) hesaplaması.

Neden quaternion? RPY (Euler açıları) sezgisel ama iki temel sorunu var:
  1. Gimbal lock: belirli konfigürasyonlarda bir eksen serbestliğini kaybeder.
  2. İnterpolasyon (SLERP gibi) doğal olarak tanımlı değildir.
Quaternion (birim, 4 boyutlu: w,x,y,z -- "scalar-first" konvansiyon) bu
sorunları çözer ve IK'nın orientation hatasını küçük bir açısal vektöre
(3 boyutlu) çevirmesini kolaylaştırır.

Quaternion konvansiyonu: q = [w, x, y, z], q = w + x*i + y*j + z*k, birim
quaternion (||q|| = 1) bir rotasyonu temsil eder.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Rotation matrix <-> RPY (Roll-Pitch-Yaw, "XYZ" sırasıyla eksen etrafında
# sabit-eksen / extrinsic; R = Rz(yaw) @ Ry(pitch) @ Rx(roll) konvansiyonu --
# rotations.py'deki Rx/Ry/Rz ile tutarlı).
# ---------------------------------------------------------------------------

def rpy_to_rotation_matrix(roll, pitch, yaw):
    """roll (X), pitch (Y), yaw (Z) radyan -> 3x3 rotasyon matrisi.
    Uygulama sırası: önce roll (X etrafında), sonra pitch (Y), sonra yaw (Z),
    hepsi SABİT (dünya/base) eksenlere göre -- yani R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    from src.kinematics.rotations import rot_x, rot_y, rot_z
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rotation_matrix_to_rpy(R):
    """3x3 rotasyon matrisinden (roll, pitch, yaw) çıkarır (yukarıdaki
    R = Rz(yaw) Ry(pitch) Rx(roll) konvansiyonunun tersi).
    Gimbal lock (pitch = +-90 derece) durumunda roll keyfi 0 alınır."""
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if np.isclose(np.cos(pitch), 0.0, atol=1e-8):
        # Gimbal lock: roll ve yaw birbirine karışır, roll=0 varsayıp yaw'ı çöz.
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)


# ---------------------------------------------------------------------------
# Rotation matrix <-> quaternion
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """3x3 rotasyon matrisi -> birim quaternion [w, x, y, z].
    Shepperd'ın sayısal olarak kararlı yöntemi (trace'e göre 4 durum)."""
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q):
    """Birim quaternion [w,x,y,z] -> 3x3 rotasyon matrisi."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),       1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),       2 * (y * z + x * w),     1 - 2 * (x ** 2 + y ** 2)],
    ])


# ---------------------------------------------------------------------------
# Quaternion cebiri
# ---------------------------------------------------------------------------

def quaternion_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quaternion_conjugate(q):
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([w, -x, -y, -z])


def quaternion_multiply(q1, q2):
    """Hamilton çarpımı: q1 * q2 (önce q2 rotasyonu, sonra q1 uygulanır)."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rpy_to_quaternion(roll, pitch, yaw):
    return rotation_matrix_to_quaternion(rpy_to_rotation_matrix(roll, pitch, yaw))


def quaternion_to_rpy(q):
    return rotation_matrix_to_rpy(quaternion_to_rotation_matrix(q))


# ---------------------------------------------------------------------------
# Orientation hatası (IK için) ve SLERP (trajectory için)
# ---------------------------------------------------------------------------

def orientation_error(R_current, R_target):
    """
    İki rotasyon matrisi arasındaki açısal hatayı 3 boyutlu bir "eksen*açı"
    vektörü olarak döndürür -- bu vektör küçükken, Jacobian'ın açısal
    kısmıyla (J_omega) doğrudan orantılıdır (IK'da e_orientation kullanılır).

    Yöntem: q_err = q_target * conjugate(q_current); hatayı q_err'ün
    vektör kısmının 2 katı olarak al (küçük açı yaklaşımı; tam açı için
    2*atan2(|v|, w)*v/|v| kullanılır, burada onu da uyguluyoruz).
    En kısa yoldan (180 derece'den fazla dönmemek için) w negatifse
    quaternion'u işaret değiştiriyoruz.
    """
    q_current = rotation_matrix_to_quaternion(R_current)
    q_target = rotation_matrix_to_quaternion(R_target)

    q_err = quaternion_multiply(q_target, quaternion_conjugate(q_current))
    if q_err[0] < 0.0:
        q_err = -q_err  # en kısa yol (shortest path)

    w = np.clip(q_err[0], -1.0, 1.0)
    v = q_err[1:]
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(v_norm, w)
    axis = v / v_norm
    return angle * axis


def quaternion_error_norm(R_current, R_target):
    """orientation_error'ın normu -- radyan cinsinden toplam açısal hata."""
    return float(np.linalg.norm(orientation_error(R_current, R_target)))


def quaternion_slerp(q0, q1, t):
    """
    Spherical Linear Interpolation: iki birim quaternion arasında, sabit
    açısal hızla enterpolasyon yapar (t in [0,1]). Doğrusal (lerp) enterpolasyon
    tercih edilmez çünkü açısal hız sabit kalmaz ve normalize etmek gerekir;
    SLERP büyük dönüşlerde de düzgün (constant angular velocity) sonuç verir.
    """
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        # En kısa yoldan git (iki quaternion aynı rotasyonu temsil eder,
        # işareti ters olan yol daha uzun olabilir).
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Neredeyse aynı yöndeler -- sayısal kararlılık için lineer enterpolasyon
        # yeterli (SLERP formülü sin(theta)'ya böler, theta->0 iken kararsız).
        result = q0 + t * (q1 - q0)
        return quaternion_normalize(result)

    theta_0 = np.arccos(dot)
    theta = theta_0 * t

    q2 = quaternion_normalize(q1 - q0 * dot)
    return q0 * np.cos(theta) + q2 * np.sin(theta)
