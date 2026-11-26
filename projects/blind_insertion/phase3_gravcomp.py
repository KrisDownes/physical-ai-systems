"""Controller-legal Gravity Compensation & GravComp-TL helper for Phase 3 (zero privileged info).

Uses an isolated arm-only MuJoCo model/data copy compiled directly from Menagerie `so101.xml`
(nominal SO-101 arm kinematics & link masses ONLY — zero peg body, zero socket body, zero table,
zero contacts). Computes exact static gravity torque `tau_grav(q_quantized)` via `mj_rne` at
zero velocity/acceleration in < 0.02 ms/step, and provides:
  1. `compute_arm_gravity_torque(q_quantized)` -> 5-DoF arm gravity torque vector (N·m)
  2. `compute_gravcomp_torque_limits_nm(q_quantized, alpha)` -> 5-DoF STS3215 Torque_Limit
     vector: clip(|tau_grav_i(q_quant)| + alpha_i * TAU_MARGIN_MAX_NM, GRAVCOMP_MIN_LIM_NM, 2.94)
  3. `GravCompBlindInsertionEnv(BlindInsertionEnv)`: thin subclass that applies GravComp-TL
     limits to `model.actuator_forcerange[:5]` without touching `env.py`, `geometry.py`, or
     `benchmarks.py`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .env import (
    GRIPPER_CLOSED_HOLD_RAD,
    MAX_TASK_DELTA_XY_M,
    MAX_TASK_DELTA_Z_M,
    PHYSICS_TIMESTEP,
    SO101_ASSETS_DIR,
    SO101_XML_PATH,
    STS3215_MAX_TORQUE_NM,
    BlindInsertionEnv,
)

TAU_MARGIN_MAX_NM = 1.50
GRAVCOMP_ALPHA_MIN = 0.04
GRAVCOMP_ALPHA_MAX = 1.00
GRAVCOMP_MIN_LIM_NM = 0.05  # minimum real STS3215 Torque_Limit (~1.7% of 2.94 Nm)

_ARM_ONLY_MODEL_CACHE: mujoco.MjModel | None = None
_ARM_ONLY_DATA_CACHE: mujoco.MjData | None = None


def _get_arm_only_model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build and cache an isolated arm-only SO-101 MuJoCo model/data (zero peg/socket/table)."""
    global _ARM_ONLY_MODEL_CACHE, _ARM_ONLY_DATA_CACHE
    if _ARM_ONLY_MODEL_CACHE is None or _ARM_ONLY_DATA_CACHE is None:
        root = ET.fromstring(SO101_XML_PATH.read_text())
        compiler = root.find('compiler')
        assert compiler is not None
        compiler.set('meshdir', str(SO101_ASSETS_DIR.resolve()))
        opt = root.find('option')
        assert opt is not None
        opt.set('timestep', f'{PHYSICS_TIMESTEP:.6g}')
        xml = ET.tostring(root, encoding='unicode')
        _ARM_ONLY_MODEL_CACHE = mujoco.MjModel.from_xml_string(xml)
        _ARM_ONLY_DATA_CACHE = mujoco.MjData(_ARM_ONLY_MODEL_CACHE)
    return _ARM_ONLY_MODEL_CACHE, _ARM_ONLY_DATA_CACHE


def compute_arm_gravity_torque(q_quantized: np.ndarray) -> np.ndarray:
    """Controller-legal nominal arm gravity torque tau_grav(q_quantized) via mj_rne (zero vel/acc).

    Strictly depends ONLY on `q_quantized` (6-DoF or 5-DoF arm angles) using an isolated arm-only
    MuJoCo model that contains ZERO peg mass, fixture, table, or contact information.
    """
    m, d = _get_arm_only_model_and_data()
    q_in = np.asarray(q_quantized, dtype=float).reshape(-1)
    if q_in.size == 5:
        d.qpos[:5] = q_in
        d.qpos[5] = GRIPPER_CLOSED_HOLD_RAD
    elif q_in.size >= 6:
        d.qpos[:6] = q_in[:6]
    else:
        raise ValueError(f'Expected at least 5 joint angles, got shape {q_in.shape}')
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    # Kinematics + COM positions + RNE (flg_acc=0, qvel=0 -> pure gravitational torque g(q))
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    tau_out = np.zeros(m.nv, dtype=float)
    mujoco.mj_rne(m, d, 0, tau_out)
    return tau_out[:5].copy()


def expand_alpha_to_5dof(alpha: float | np.ndarray) -> np.ndarray:
    """Expand scalar, 2-group [alpha_lat, alpha_arm], or 5-DoF alpha into 5-DoF clipped vector."""
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
    """Compute per-joint STS3215 GravComp-TL limits (N·m) and nominal tau_grav(q_quantized).

    lim_i = clip(|tau_grav_i(q_quant)| + alpha_i * tau_margin_max_nm, min_lim_nm, 2.94 N·m)
    """
    tau_g = compute_arm_gravity_torque(q_quantized)
    a5 = expand_alpha_to_5dof(alpha)
    lim_nm = np.clip(np.abs(tau_g) + a5 * float(tau_margin_max_nm), float(min_lim_nm), STS3215_MAX_TORQUE_NM)
    return lim_nm, tau_g


class GravCompBlindInsertionEnv(BlindInsertionEnv):
    """Thin subclass of BlindInsertionEnv adding real-hardware GravComp-TL mode without modifying env.py.

    Accepts either standard BlindInsertionEnv actions (delegated identically) OR a dict:
      {'delta_xyz': [dx, dy, dz], 'gravcomp_alpha': scalar | [a_lat, a_arm] | 5-vec}
    which computes controller-legal `lim_i = clip(|tau_grav_i(q_quant)| + alpha_i * 1.50, 0.05, 2.94)`
    and sets `model.actuator_forcerange[i] = [-lim_i, +lim_i]` directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_gravcomp_lim_nm: np.ndarray | None = None

    def _apply_torque_limits(self, torque_scale: np.ndarray) -> None:
        if self._pending_gravcomp_lim_nm is not None:
            lims = np.clip(self._pending_gravcomp_lim_nm, GRAVCOMP_MIN_LIM_NM, STS3215_MAX_TORQUE_NM)
            self._pending_gravcomp_lim_nm = None
            self._torque_scale = lims / STS3215_MAX_TORQUE_NM
            for j in range(5):
                lim = float(lims[j])
                self.model.actuator_forcerange[j, 0] = -lim
                self.model.actuator_forcerange[j, 1] = lim
            self.model.actuator_forcerange[5, 0] = -STS3215_MAX_TORQUE_NM
            self.model.actuator_forcerange[5, 1] = STS3215_MAX_TORQUE_NM
        else:
            super()._apply_torque_limits(torque_scale)

    def _parse_action(self, action, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(action, dict) and 'gravcomp_alpha' in action:
            d_xyz = np.asarray(action['delta_xyz'], dtype=float)
            dq = self._task_delta_to_joint_delta(q_ref, d_xyz)
            lim_nm, _ = compute_gravcomp_torque_limits_nm(q_ref, action['gravcomp_alpha'])
            self._pending_gravcomp_lim_nm = lim_nm
            ts = lim_nm / STS3215_MAX_TORQUE_NM
            prev_act = np.concatenate([
                [np.clip(d_xyz[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
                 np.clip(d_xyz[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
                 np.clip(d_xyz[2], -MAX_TASK_DELTA_Z_M, MAX_TASK_DELTA_Z_M)],
                ts,
            ])
            return dq, ts, prev_act
        self._pending_gravcomp_lim_nm = None
        return super()._parse_action(action, q_ref)
