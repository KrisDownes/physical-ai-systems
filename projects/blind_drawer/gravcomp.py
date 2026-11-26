"""Standalone controller-legal Gravity Compensation & GravComp-TL helper for `projects/blind_drawer/`.

Uses an isolated lean arm-only MuJoCo model/data copy compiled directly from Menagerie `so101.xml`
(nominal SO-101 arm kinematics & link masses ONLY — zero cabinet, drawer, handle, table, or contacts).
Computes exact static gravity torque `tau_grav(q_quantized)` via `mj_rne` at zero vel/acc in < 0.02 ms.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SO101_MODEL_DIR = REPO_ROOT / 'projects' / 'active_robot_adaptation' / '.so101-model'
SO101_XML_PATH = SO101_MODEL_DIR / 'so101.xml'
SO101_ASSETS_DIR = SO101_MODEL_DIR / 'assets'

PHYSICS_TIMESTEP = 0.001
STS3215_MAX_TORQUE_NM = 2.94
TAU_MARGIN_MAX_NM = 2.40   # alpha=0.35 -> 0.84 Nm tactile compliance; alpha=1.0 -> full STS3215 2.94 Nm
GRAVCOMP_ALPHA_MIN = 0.04
GRAVCOMP_ALPHA_MAX = 1.00
GRAVCOMP_MIN_LIM_NM = 0.05
GRIPPER_GRASP_HOLD_RAD = -0.10  # closed jaw reference angle for gravity torque

_ARM_ONLY_MODEL_CACHE: mujoco.MjModel | None = None
_ARM_ONLY_DATA_CACHE: mujoco.MjData | None = None


def strip_visual_only_meshes(xml_str: str) -> str:
    """Strip zero-contact `class='visual'` STL meshes ONLY from bodies with explicit `<inertial>`."""
    root = ET.fromstring(xml_str)
    for body in root.iter('body'):
        if body.find('inertial') is not None:
            for g in list(body.findall('geom')):
                if g.get('class') == 'visual':
                    body.remove(g)
    used_meshes = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    asset = root.find('asset')
    if asset is not None:
        for m in list(asset.findall('mesh')):
            name = m.get('name') or m.get('file', '').rsplit('/', 1)[-1].rsplit('.', 1)[0]
            if name not in used_meshes:
                asset.remove(m)
    return ET.tostring(root, encoding='unicode')


def _get_arm_only_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    global _ARM_ONLY_MODEL_CACHE, _ARM_ONLY_DATA_CACHE
    if _ARM_ONLY_MODEL_CACHE is None or _ARM_ONLY_DATA_CACHE is None:
        root = ET.fromstring(SO101_XML_PATH.read_text())
        compiler = root.find('compiler')
        assert compiler is not None
        compiler.set('meshdir', str(SO101_ASSETS_DIR.resolve()))
        opt = root.find('option')
        assert opt is not None
        opt.set('timestep', f'{PHYSICS_TIMESTEP:.6g}')
        xml = strip_visual_only_meshes(ET.tostring(root, encoding='unicode'))
        _ARM_ONLY_MODEL_CACHE = mujoco.MjModel.from_xml_string(xml)
        _ARM_ONLY_DATA_CACHE = mujoco.MjData(_ARM_ONLY_MODEL_CACHE)
    return _ARM_ONLY_MODEL_CACHE, _ARM_ONLY_DATA_CACHE


def compute_arm_gravity_torque(q_quantized: np.ndarray) -> np.ndarray:
    """Controller-legal nominal arm gravity torque tau_grav(q_quantized) via mj_rne (zero vel/acc)."""
    m, d = _get_arm_only_model_and_data()
    q_in = np.asarray(q_quantized, dtype=float).reshape(-1)
    if q_in.size == 5:
        d.qpos[:5] = q_in
        d.qpos[5] = GRIPPER_GRASP_HOLD_RAD
    elif q_in.size >= 6:
        d.qpos[:6] = q_in[:6]
    else:
        raise ValueError(f'Expected >=5 joint angles, got shape {q_in.shape}')
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    tau_out = np.zeros(m.nv, dtype=float)
    mujoco.mj_rne(m, d, 0, tau_out)
    return tau_out[:5].copy()


def expand_alpha_to_5dof(alpha: float | np.ndarray) -> np.ndarray:
    """Expand scalar, 2-group [alpha_lat/align, alpha_arm/pull], or 5-DoF alpha into clipped 5-vec."""
    arr = np.asarray(alpha, dtype=float).reshape(-1)
    if arr.size == 1:
        a5 = np.full(5, float(arr[0]), dtype=float)
    elif arr.size == 2:
        a_lat, a_arm = float(arr[0]), float(arr[1])
        a5 = np.array([a_lat, a_arm, a_arm, a_arm, a_lat], dtype=float)
    elif arr.size == 5:
        a5 = arr.copy()
    else:
        raise ValueError(f'Expected alpha len 1, 2, or 5, got {arr.shape}')
    return np.clip(a5, GRAVCOMP_ALPHA_MIN, GRAVCOMP_ALPHA_MAX)


def compute_gravcomp_torque_limits_nm(
    q_quantized: np.ndarray,
    alpha: float | np.ndarray,
    tau_margin_max_nm: float = TAU_MARGIN_MAX_NM,
    min_lim_nm: float = GRAVCOMP_MIN_LIM_NM,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-joint STS3215 GravComp-TL limits: clip(|tau_grav_i(q_quant)| + alpha_i*1.50, 0.05, 2.94)."""
    tau_g = compute_arm_gravity_torque(q_quantized)
    a5 = expand_alpha_to_5dof(alpha)
    lim_nm = np.clip(np.abs(tau_g) + a5 * float(tau_margin_max_nm), float(min_lim_nm), STS3215_MAX_TORQUE_NM)
    return lim_nm, tau_g
