"""
Multiple IK Solutions: 6-DOF bir kolda aynı hedefe farklı eklem
konfigürasyonlarıyla ulaşılabilir (örn. "dirsek yukarı" / "dirsek aşağı").
Tek bir başlangıç tahmininden (q_init) yola çıkan gradyan-tabanlı bir IK
çözücü (DLS gibi), o başlangıca en yakın YEREL çözüme yakınsar -- global
olarak "en iyi" çözümü garanti etmez.

Bu modül, FARKLI başlangıç noktalarından ("seed") aynı hedefi birden çok kez
çözüp, sonuçları ağırlıklı bir maliyet fonksiyonuyla kıyaslayarak en iyisini
seçer:

    cost = w1 * E_pose(mm) + w2 * E_joint_motion(derece) + w3 * (1 / manipulability)

- E_pose: hedefe ne kadar yakın ulaşıldığı (küçük olsun isteriz)
- E_joint_motion: mevcut konumdan (q_current) ne kadar hareket gerektiği
  (küçük olsun isteriz -- daha az hareket = daha güvenli/hızlı)
- 1/manipulability: singularity'ye ne kadar yakın olunduğu (manipulability
  küçükse bu terim büyür, yani singularity'ye yakın çözümler cezalandırılır)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np

from src.kinematics.dh import UR5_DH_PARAMS
from src.kinematics.jacobian import compute_jacobian
from src.analysis.manipulability import yoshikawa_manipulability
from src.kinematics.ik_dls import inverse_kinematics_dls
from src.robot.joint_limits import JOINT_LIMITS


@dataclass(eq=False)
class IKCandidate:
    q: np.ndarray
    converged: bool
    pose_error_mm: float
    joint_motion_deg: float
    manipulability: float
    cost: float
    label: str = ""


@dataclass(eq=False)
class MultiStartResult:
    selected: IKCandidate
    candidates: List[IKCandidate] = field(default_factory=list)


def generate_seed_configs(n_joints: int = 6, n_random: int = 5, joint_limits=JOINT_LIMITS, seed: int = 0):
    """
    Birkaç "yapısal" başlangıç noktası (sıfır, ve eklem 2/3'ü zıt işaretlerde
    -- kabaca 'dirsek yukarı'/'dirsek aşağı' hissi veren konfigürasyonlar) artı
    birkaç rastgele başlangıç noktası üretir. Amaç, farklı yerel minimumlara
    düşecek çeşitlilikte q_init kümesi sağlamak.
    """
    seeds = [
        ("zero", np.zeros(n_joints)),
        ("elbow-up-ish", np.array([0.0, -0.5, 1.0, 0.0, 0.5, 0.0])[:n_joints]),
        ("elbow-down-ish", np.array([0.0, 0.5, -1.0, 0.0, -0.5, 0.0])[:n_joints]),
    ]
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        q_rand = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits[:n_joints]])
        seeds.append((f"random-{i}", q_rand))
    return seeds


def multi_start_ik(
    target_position,
    target_orientation=None,
    q_current=None,
    dh_params=UR5_DH_PARAMS,
    solver: Callable = inverse_kinematics_dls,
    n_random_seeds: int = 5,
    weights=(1.0, 0.01, 0.1),
    joint_limits=JOINT_LIMITS,
    n_joints: int = 6,
    seed: int = 0,
    **solver_kwargs,
) -> MultiStartResult:
    """
    Birden fazla başlangıç noktasından `solver` ile IK çözer, her sonucu
    puanlar ve en düşük maliyetli (en iyi) çözümü seçer.

    `q_current`, joint_motion maliyetini hesaplamak için kullanılır (None ise
    sıfır pozisyonu varsayılır).
    """
    w_pose, w_motion, w_singularity = weights
    q_current = np.array(q_current, dtype=float) if q_current is not None else np.zeros(n_joints)

    candidates: List[IKCandidate] = []
    for label, q_init in generate_seed_configs(n_joints, n_random_seeds, joint_limits, seed):
        result = solver(
            target_position=target_position, target_orientation=target_orientation,
            q_init=q_init, dh_params=dh_params, joint_limits=joint_limits,
            n_joints=n_joints, **solver_kwargs,
        )
        pose_error_mm = result.position_error * 1000.0
        joint_motion_deg = float(np.degrees(np.linalg.norm(result.q - q_current)))
        J = compute_jacobian(result.q, dh_params)
        manip = yoshikawa_manipulability(J)

        cost = (
            w_pose * pose_error_mm
            + w_motion * joint_motion_deg
            + w_singularity * (1.0 / max(manip, 1e-6))
        )
        if not result.converged:
            cost += 1e6  # yakınsamayan çözümleri ağır cezalandır (pratikte elenir)

        candidates.append(IKCandidate(
            q=result.q, converged=result.converged, pose_error_mm=pose_error_mm,
            joint_motion_deg=joint_motion_deg, manipulability=manip, cost=cost, label=label,
        ))

    best = min(candidates, key=lambda c: c.cost)
    return MultiStartResult(selected=best, candidates=candidates)
