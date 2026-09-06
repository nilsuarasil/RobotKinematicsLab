"""
Basit workspace / reachability kontrolü.

Not: Bu sadece robot tabanından öklidyen uzaklığa bakan kaba bir kontrol.
IK çözücü zaten başarısız olursa "ulaşılamaz" diyebilir, ama IK'yı hiç
çalıştırmadan önce ucuz bir ön-kontrol yapmak, hem hızlı hem de kullanıcıya
daha net bir mesaj vermek için faydalı.
"""
import numpy as np

# UR5 resmi spesifikasyonuna göre maksimum erişim ~850 mm.
UR5_MAX_REACH = 0.850
# Robotun kendi tabanına çok yakın (link çakışması / singularity riski
# taşıyan) noktalara erişimi sınırlamak için küçük bir minimum erişim.
UR5_MIN_REACH = 0.05


def is_reachable(target, base_position=(0.0, 0.0, 0.0),
                  max_reach=UR5_MAX_REACH, min_reach=UR5_MIN_REACH):
    """
    Hedefin robot tabanından uzaklığının [min_reach, max_reach] aralığında
    olup olmadığını kontrol eder. Döndürür: (bool reachable, float distance).
    """
    target = np.asarray(target, dtype=float)
    base_position = np.asarray(base_position, dtype=float)
    distance = float(np.linalg.norm(target - base_position))
    reachable = min_reach <= distance <= max_reach
    return reachable, distance
