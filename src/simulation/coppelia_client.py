"""
CoppeliaSim ZeroMQ Remote API istemcisi.

Kullanmadan önce:
1. CoppeliaSim'i aç (Student/Edu sürümü yeterli), sahnene Model Browser'dan
   `robots > non-mobile > UR5` modelini ekle.
2. `pip install coppeliasim-zmqremoteapi-client`
3. CoppeliaSim >= 4.4, ZMQ remote API sunucusunu varsayılan olarak
   localhost:23000 üzerinde otomatik başlatır.

Önemli detay -- CoppeliaSim'in standart UR5 modelinde 6 eklemin de sahne
hiyerarşisindeki adı birebir aynıdır ("joint"), her biri bir öncekinin
içine iç içe geçmiş haldedir (joint -> link -> joint -> link -> ...).
Yani isimle tek tek adreslemek (`/UR5/joint1` gibi) bu modelde ÇALIŞMAZ --
hepsi "joint" ismini taşıdığı için isim çakışması olur.

Bunun yerine `sim.getObjectsInTree` ile robotun tüm alt ağacını tarayıp,
tipi "joint" olan objeleri sırayla topluyoruz. Zincir dallanmasız (seri,
tek çocuklu) bir yapı olduğu için bu tarama otomatik olarak base->tip
sırasını (joint1..joint6) doğru veriyor. End-effector için ise "connection"
adlı obje kullanılıyor (sahnede tek bir tane var, o yüzden doğrudan adıyla
bulunabiliyor).

NOT: Bu modül, bulut ortamında GUI simülatörü çalıştırılamadığı için gerçek
bir CoppeliaSim örneğine bağlanılarak test EDİLMEDİ; yukarıdaki tasarım
kullanıcının paylaştığı Scene Hierarchy ekran görüntüsüne göre yapıldı.
Farklı bir sahne/robot kullanıyorsan `joint_names` parametresiyle elle
isim listesi vererek eski (isimle adresleme) davranışına dönebilirsin.
"""
import time
from typing import List, Optional
import numpy as np

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError:
    RemoteAPIClient = None  # Kütüphane kurulu değilse import zamanında patlamasın


class CoppeliaSimClient:
    def __init__(self, joint_names: Optional[List[str]] = None,
                 robot_base_name: str = "/UR5",
                 end_effector_name: str = "/connection",
                 host: str = "localhost", port: int = 23000,
                 stepping: bool = True):
        if RemoteAPIClient is None:
            raise ImportError(
                "coppeliasim-zmqremoteapi-client kurulu değil. "
                "Kur: pip install coppeliasim-zmqremoteapi-client"
            )
        self.stepping = stepping

        self.client = RemoteAPIClient(host=host, port=port)
        self.sim = self.client.require('sim')

        self.robot_handle = None
        if joint_names is not None:
            # Elle verilen isim listesi (farklı/özel isimlendirilmiş sahneler için)
            self.joint_handles = [self.sim.getObject(name) for name in joint_names]
        else:
            # Otomatik keşif: robotun alt ağacındaki tüm joint-tipi objeleri
            # hiyerarşi sırasına göre topla (bkz. yukarıdaki modül notu).
            self.robot_handle = self.sim.getObject(robot_base_name)
            self.joint_handles = self.sim.getObjectsInTree(
                self.robot_handle, self.sim.object_joint_type, 0
            )
            if len(self.joint_handles) != 6:
                raise RuntimeError(
                    f"'{robot_base_name}' altında 6 joint bekleniyordu, "
                    f"{len(self.joint_handles)} bulundu. Sahnen farklıysa "
                    f"joint_names parametresiyle elle isim listesi ver."
                )

        self.end_effector_handle = self.sim.getObject(end_effector_name)

    def start_simulation(self):
        """
        Simülasyonu başlatır. `stepping` modu burada açılıyor (constructor'da
        DEĞİL) çünkü stepping, CoppeliaSim'in TÜM simülasyon döngüsünü dışarıdan
        (bu script'ten) kontrol edilir hale getiriyor -- eğer script stepping'i
        açıp sonra (hata/erken çıkış yüzünden) hiç kapatmadan bağlantıyı
        bırakırsa, CoppeliaSim "dışarıdan adım bekliyor ama kimse adım vermiyor"
        durumunda kilitli kalıyor (arayüzde "RUNNING" yazıp donmuş gibi görünür).
        Bu yüzden start/stop_simulation'ı HER ZAMAN try/finally ile birlikte
        kullan (bkz. main.py) ki stop_simulation mutlaka çağrılsın.
        """
        if self.stepping:
            self.client.setStepping(True)
        self.sim.startSimulation()

    def stop_simulation(self):
        self.sim.stopSimulation()
        if self.stepping:
            # Stepping modunu mutlaka kapat -- CoppeliaSim'in kendi Play/Stop
            # butonlarının tekrar normal (gerçek zamanlı) çalışabilmesi için şart.
            self.client.setStepping(False)

    def set_joint_positions(self, q):
        """
        Eklem açılarını (radyan) hedef pozisyon olarak gönderir. Robot
        dinamiği aktifse motor kontrolüyle o pozisyona hareket eder;
        kinematik modda ise anlık atlar -- bu yüzden ani bir `set` yerine
        `send_trajectory` ile ara noktalar göndermek daha gerçekçi bir
        hareket verir.
        """
        for handle, angle in zip(self.joint_handles, q):
            self.sim.setJointTargetPosition(handle, float(angle))

    def get_joint_positions(self):
        return np.array([self.sim.getJointPosition(h) for h in self.joint_handles])

    def get_end_effector_position(self):
        pos = self.sim.getObjectPosition(self.end_effector_handle, self.sim.handle_world)
        return np.array(pos)

    def get_base_position(self):
        """Robotun taban objesinin dünya koordinatındaki konumu (varsa)."""
        if self.robot_handle is None:
            return None
        return np.array(self.sim.getObjectPosition(self.robot_handle, self.sim.handle_world))

    def get_base_pose(self):
        """
        Robotun taban objesinin dünya çerçevesindeki TAM pose'u: 4x4
        homogeneous transform (rotasyon + öteleme). Sadece konum değil,
        robotun sahnede hangi yöne döndürülmüş olduğunu da yakalar --
        `sim.getObjectMatrix` 3x4'lük bir dönüşüm matrisi döndürür (satır
        başına düz [r,r,r,t] -- son sütun öteleme).
        """
        if self.robot_handle is None:
            return None
        m = np.array(self.sim.getObjectMatrix(self.robot_handle, self.sim.handle_world), dtype=float)
        T = np.eye(4)
        T[:3, :4] = m.reshape(3, 4)
        return T

    def calibrate_local_transforms(self):
        """
        Sahnenin MEVCUT durumundan (tüm eklemler q=0 iken -- taze bir sahne,
        henüz hiç hareket ettirilmemişken çağır) her eklemin bir öncekine
        göre GERÇEK statik montaj transformunu okuyup, DH tablosu YERİNE
        kullanılacak bir kinematik zincir çıkarır.

        Neden gerekli? Literatür DH parametreleri (kinematics/dh.py) bazı
        CoppeliaSim UR5 modelleriyle "sıfır açı" referansı bakımından
        örtüşmeyebilir (bkz. main.py'deki [kalibrasyon] teşhis çıktısı).
        Bu fonksiyon, geometriyi TAHMİN etmek yerine doğrudan sahneden
        ölçerek bu sorunu kökten çözer -- CoppeliaSim'in joint açısını
        `staticMontaj @ Rz(theta)` şeklinde (montaj transformunun ÜZERİNE,
        eklemin kendi yerel Z'si etrafında) uyguladığı bilgisine dayanır.

        Döndürür: {"base_transform", "joint_local_transforms" (6 adet 4x4),
        "tip_offset"} -- bkz. robot_model.CalibratedUR5Robot.
        """
        def world_matrix(handle):
            m = np.array(self.sim.getObjectMatrix(handle, self.sim.handle_world), dtype=float)
            T = np.eye(4)
            T[:3, :4] = m.reshape(3, 4)
            return T

        base_T = world_matrix(self.robot_handle) if self.robot_handle is not None else np.eye(4)
        joint_world_transforms = [world_matrix(h) for h in self.joint_handles]
        ee_world_T = world_matrix(self.end_effector_handle)

        joint_local_transforms = []
        prev_T = base_T
        for T_world in joint_world_transforms:
            joint_local_transforms.append(np.linalg.inv(prev_T) @ T_world)
            prev_T = T_world
        tip_offset = np.linalg.inv(prev_T) @ ee_world_T

        return {
            "base_transform": base_T,
            "joint_local_transforms": joint_local_transforms,
            "tip_offset": tip_offset,
        }

    def send_trajectory(self, trajectory, step_delay: float = 0.02, settle_steps: int = 100):
        """
        trajectory: shape (steps, n_joints) -- planning/trajectory.py'den
        gelen ara noktalar. Her adımda hedefleri gönderip simülasyonu bir
        adım (stepping modundaysa `client.step()`, değilse gerçek zamanlı
        `time.sleep`) ilerletir.

        Eklemler dinamik/motor kontrollü olduğu için `setJointTargetPosition`
        çağrısı motoru anında o açıya atlatmaz -- zamanla (PID ile) oraya
        yaklaşır. Bu yüzden trajectory bittikten sonra, son hedefte
        `settle_steps` kadar ekstra simülasyon adımı daha atarak motorların
        gerçekten oturmasına (steady-state'e ulaşmasına) izin veriyoruz.
        """
        for q_step in trajectory:
            self.set_joint_positions(q_step)
            if self.stepping:
                self.client.step()
            else:
                time.sleep(step_delay)

        if len(trajectory) > 0 and settle_steps > 0:
            final_q = trajectory[-1]
            for _ in range(settle_steps):
                self.set_joint_positions(final_q)
                if self.stepping:
                    self.client.step()
                else:
                    time.sleep(step_delay)
