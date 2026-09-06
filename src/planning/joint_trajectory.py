"""
V2 repo mimarisinde eklem-uzayı trajectory'leri bu modülün altında adlandırılıyor
(bkz. proje mimarisi notları). Gerçek implementasyon `trajectory.py`'de kalmaya
devam ediyor (V1'den beri var, testleri de ona bağlı) -- burada sadece V2'nin
beklediği isimle yeniden dışa aktarıyoruz, böylece hem eski hem yeni import
yolu çalışır.
"""
from src.planning.trajectory import linear_joint_trajectory, quintic_joint_trajectory

__all__ = ["linear_joint_trajectory", "quintic_joint_trajectory"]
