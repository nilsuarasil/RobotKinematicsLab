"""
UR5 robot modelini tek bir nesnede toplayan kolaylık sınıfı: FK, Jacobian,
IK, workspace kontrolünü robotun dünya çerçevesindeki TAM pose'una (rotasyon +
öteleme) göre sarar.

Neden sadece öteleme (position) değil, tam bir transform (base_transform)?
CoppeliaSim sahnesinde robot sürüklenip bırakıldığında hem yeri hem de yönü
(rotasyonu) dünya çerçevesine göre keyfi olabilir. Sadece "hedef - taban
konumu" şeklinde bir çıkarma yapmak, taban DÖNDÜRÜLMEMİŞSE doğru sonuç verir;
ama taban döndürülmüşse (ki CoppeliaSim'de bu oldukça olası), bu basit
çıkarma yanlış bir "lokal hedef" üretir ve IK, robotun aslında ulaşabileceği
ama farklı bir yöndeki bir noktaya ulaşacak açılar hesaplar -- joint açıları
"doğru" görünür (IK kendi içinde tutarlıdır) ama gerçek dünyada end-effector
hedeften uzak kalır. Bu yüzden burada tam bir 4x4 homogeneous transform
kullanıyoruz.
"""
import numpy as np
from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.forward import forward_kinematics
from src.kinematics.jacobian import compute_jacobian
from src.kinematics.inverse import inverse_kinematics, inverse_kinematics_generic
from src.kinematics.rotations import rot_z
from src.kinematics.transforms import homogeneous_transform, inverse_transform, transform_point
from src.kinematics.ik_pseudoinverse import inverse_kinematics_pseudoinverse
from src.kinematics.ik_dls import inverse_kinematics_dls
from src.kinematics.ik_jacobian_transpose import inverse_kinematics_jacobian_transpose
from src.kinematics.ik_multi_start import multi_start_ik
from src.kinematics._ik_common import resolve_target_rotation
from src.control.joint_limit_avoidance import inverse_kinematics_with_joint_limit_avoidance
from src.planning.workspace import is_reachable
from src.robot.joint_limits import JOINT_LIMITS

V2_SOLVERS = {
    "pseudoinverse": inverse_kinematics_pseudoinverse,
    "dls": inverse_kinematics_dls,
    "transpose": inverse_kinematics_jacobian_transpose,
    "joint_limit_avoidance": inverse_kinematics_with_joint_limit_avoidance,
}


class UR5Robot:
    def __init__(self, base_position=(0.0, 0.0, 0.0), base_rotation=None,
                 base_transform=None, dh_params=UR5_DH_PARAMS,
                 joint_limits=JOINT_LIMITS):
        """
        base_transform verilirse (4x4 numpy array) doğrudan kullanılır --
        CoppeliaSim'den okunan gerçek taban pose'unu geçirmek için idealdir.
        Verilmezse base_position (3,) ve base_rotation (3x3, verilmezse
        birim matris/rotasyonsuz) birleştirilerek oluşturulur.
        """
        if base_transform is not None:
            self.base_transform = np.asarray(base_transform, dtype=float)
        else:
            R = np.eye(3) if base_rotation is None else np.asarray(base_rotation, dtype=float)
            self.base_transform = homogeneous_transform(R, base_position)

        self.base_position = self.base_transform[:3, 3].copy()
        self.dh_params = dh_params
        self.joint_limits = joint_limits
        self.n_joints = len(dh_params)

    def forward(self, q):
        """Base transform dahil, dünya çerçevesindeki tam end-effector pose'u (4x4)."""
        T_local = forward_kinematics(q, self.dh_params)
        return self.base_transform @ T_local

    def end_effector_position(self, q):
        return self.forward(q)[:3, 3]

    def jacobian(self, q):
        # NOT: Bu, robotun KENDİ (base) çerçevesindeki Jacobian'dır. Taban
        # dönmüş olsa bile eklem hızı <-> lokal hız ilişkisi değişmez;
        # sadece dünya çerçevesine dönüştürmek istersen R_base ile çarpman
        # gerekir (V1 kapsamında buna ihtiyacımız yok).
        return compute_jacobian(q, self.dh_params)

    def is_reachable(self, target):
        return is_reachable(target, base_position=self.base_position)

    def solve_ik(self, target_position, q_init=None, **kwargs):
        # Hedefi dünya çerçevesinden robotun kendi (base) çerçevesine çevir --
        # taban rotasyonunu da hesaba katan tam ters transform ile.
        base_transform_inv = inverse_transform(self.base_transform)
        local_target = transform_point(base_transform_inv, target_position)
        return inverse_kinematics(
            local_target, q_init=q_init,
            joint_limits=self.joint_limits, n_joints=self.n_joints, **kwargs
        )

    def solve_ik_v2(self, target_position, target_orientation=None, q_init=None,
                     solver: str = "dls", use_multi_start: bool = False, **kwargs):
        """
        V2 IK: hedef pozisyon + (opsiyonel) hedef orientation'ı, seçilen
        solver ("pseudoinverse" | "dls" | "transpose") ile çözer. `target_orientation`
        3x3 rotasyon matrisi, quaternion (4,) ya da RPY (3, radyan) olabilir --
        DÜNYA çerçevesinde verilir, burada robotun (muhtemelen döndürülmüş)
        base çerçevesine çevrilir (V1'deki solve_ik ile aynı mantık, artık
        orientation'ı da kapsıyor).

        `use_multi_start=True` ise tek bir q_init yerine birden fazla başlangıç
        noktasından çözüp en iyi (pose hatası + eklem hareketi + manipulability
        ağırlıklı) sonucu seçer (bkz. ik_multi_start.py).
        """
        base_transform_inv = inverse_transform(self.base_transform)
        local_target_position = transform_point(base_transform_inv, target_position)

        local_target_orientation = None
        target_R_world = resolve_target_rotation(target_orientation)
        if target_R_world is not None:
            local_target_orientation = base_transform_inv[:3, :3] @ target_R_world

        if use_multi_start:
            solver_fn = V2_SOLVERS[solver]
            return multi_start_ik(
                target_position=local_target_position, target_orientation=local_target_orientation,
                q_current=q_init, dh_params=self.dh_params, solver=solver_fn,
                joint_limits=self.joint_limits, n_joints=self.n_joints, **kwargs
            )

        solver_fn = V2_SOLVERS[solver]
        return solver_fn(
            target_position=local_target_position, target_orientation=local_target_orientation,
            q_init=q_init, dh_params=self.dh_params,
            joint_limits=self.joint_limits, n_joints=self.n_joints, **kwargs
        )


class CalibratedUR5Robot:
    """
    DH tablosuna (yani literatürden alınan a/alpha/d değerlerine) güvenmek
    yerine, CoppeliaSim sahnesinden EMPİRİK olarak okunan gerçek eklem-eklem
    transformlarını kullanan robot modeli.

    Neden gerekli? Farklı CAD/simülatör modelleri, bir eklemin "sıfır açı"
    referans duruşunu farklı tanımlayabiliyor. Örneğin bizim UR5_DH_PARAMS
    tablomuza göre q=0 kolun YATAY uzandığı bir duruş, ama CoppeliaSim'in
    hazır UR5 modelinde q=0 kolun belirgin şekilde DİKEY/bükülü durduğu bir
    duruş olabilir -- bu iki "sıfır" birbirine denk değilse, doğru joint
    açılarını hesaplasak bile robot gerçek dünyada hedeften uzak kalır
    (bkz. main.py'deki [kalibrasyon] teşhis çıktısı).

    Bu sınıf bu sorunu tahmin/deneme-yanılma ile değil, sahnenin KENDİSİNDEN
    ölçerek çözer: her eklemin bir öncekine göre gerçek statik montaj
    transformunu (CoppeliaSim'de `sim.getObjectMatrix`) okur, sonra CoppeliaSim'in
    joint açılarını nasıl uyguladığıyla (statik montaj transformunun ÜZERİNE,
    eklemin kendi yerel Z ekseni etrafında bir Rz(theta) eklenerek) BİREBİR
    aynı matematiği kurar:

        T(q) = base_transform . T0.Rz(q0) . T1.Rz(q1) . ... . T5.Rz(q5) . tip_offset

    Bkz. CoppeliaSimClient.calibrate_local_transforms().
    """

    def __init__(self, calibration, joint_limits=JOINT_LIMITS):
        self.base_transform = np.asarray(calibration["base_transform"], dtype=float)
        self.joint_local_transforms = [
            np.asarray(T, dtype=float) for T in calibration["joint_local_transforms"]
        ]
        self.tip_offset = np.asarray(calibration["tip_offset"], dtype=float)
        self.joint_limits = joint_limits
        self.n_joints = len(self.joint_local_transforms)
        self.base_position = self.base_transform[:3, 3].copy()

    def forward(self, q):
        T = self.base_transform
        for T_local, theta in zip(self.joint_local_transforms, q):
            T = T @ T_local @ homogeneous_transform(rot_z(theta), [0.0, 0.0, 0.0])
        return T @ self.tip_offset

    def end_effector_position(self, q):
        return self.forward(q)[:3, 3]

    def is_reachable(self, target):
        return is_reachable(target, base_position=self.base_position)

    def solve_ik(self, target_position, q_init=None, **kwargs):
        # Analitik Jacobian yok (geometri koddan değil sahneden okunuyor) --
        # bu yüzden sayısal (finite-difference) Jacobian tabanlı genel IK
        # çözücüyü kullanıyoruz. Matematiği DH-tabanlı IK ile birebir aynı.
        return inverse_kinematics_generic(
            fk_func=self.end_effector_position,
            n_joints=self.n_joints,
            target_position=target_position,
            q_init=q_init,
            joint_limits=self.joint_limits,
            **kwargs
        )
