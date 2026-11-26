# SPDX-License-Identifier: MIT
"""Core Gravity-Compensated Servo Torque Limiting (GravComp-TL) controller for SO-101 / STS3215.

Core per-joint law (i = 0..4: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll):

    tau_lim_i  = clip(|tau_grav_i(q_quant)| + alpha_i * tau_margin_max, tau_min, 2.94 N·m)
    reg_0x30_i = round(1000 * tau_lim_i / 2.94)
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SO101_DIR = REPO_ROOT / 'projects' / 'active_robot_adaptation' / '.so101-model'

JOINT_NAMES: tuple[str, ...] = (
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
)


@dataclass(frozen=True)
class STS3215Spec:
    """Feetech STS3215 (7.4 V) hardware & register specification."""
    encoder_bits: int = 12
    encoder_ticks: int = 4096
    stall_torque_nm: float = 2.94           # 30 kg·cm at 7.4 V
    goal_position_addr: int = 0x2A          # 2-byte Goal_Position [0..4095]
    torque_limit_addr: int = 0x30           # 2-byte Torque_Limit [0..1000]
    present_position_addr: int = 0x38       # 2-byte Present_Position [0..4095]
    present_load_addr: int = 0x3C           # 2-byte Present_Load (sign-magnitude [0..1000])
    torque_limit_reg_max: int = 1000
    tau_margin_max_nm: float = 2.40
    min_torque_lim_nm: float = 0.05
    alpha_min: float = 0.05
    alpha_max: float = 1.00
    nominal_backlash_rad: float = 0.0045    # ~3 encoder counts measured gear backlash

    @property
    def rad_per_tick(self) -> float:
        return 2.0 * math.pi / float(self.encoder_ticks)


DEFAULT_SPEC = STS3215Spec()


def strip_visual_only_meshes(xml_str: str) -> str:
    """Strip zero-contact `class='visual'` STL meshes from bodies with explicit `<inertial>`."""
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


def expand_alpha(alpha: float | np.ndarray, n_joints: int = 5, spec: STS3215Spec = DEFAULT_SPEC) -> np.ndarray:
    """Expand scalar, 2-group [alpha_lat, alpha_arm], or N-DoF alpha into clipped (N,) array."""
    arr = np.asarray(alpha, dtype=float).reshape(-1)
    if arr.size == 1:
        a = np.full(n_joints, float(arr[0]), dtype=float)
    elif arr.size == 2 and n_joints == 5:
        a = np.array([arr[0], arr[1], arr[1], arr[1], arr[0]], dtype=float)
    elif arr.size == n_joints:
        a = arr.copy()
    else:
        raise ValueError(f'Expected alpha of length 1, 2, or {n_joints}; got shape {arr.shape}')
    return np.clip(a, spec.alpha_min, spec.alpha_max)


def nm_to_sts3215_ticks(lim_nm: np.ndarray, spec: STS3215Spec = DEFAULT_SPEC) -> np.ndarray:
    """Map physical torque limits (N·m) to STS3215 `0x30` Torque_Limit register ticks [0..1000]."""
    frac = np.clip(np.asarray(lim_nm, dtype=float) / spec.stall_torque_nm, 0.0, 1.0)
    return np.rint(frac * spec.torque_limit_reg_max).astype(np.int32)


def sts3215_ticks_to_nm(reg_ticks: np.ndarray, spec: STS3215Spec = DEFAULT_SPEC) -> np.ndarray:
    """Map STS3215 `0x30` Torque_Limit register ticks [0..1000] back to N·m."""
    reg = np.clip(np.asarray(reg_ticks, dtype=float), 0.0, float(spec.torque_limit_reg_max))
    return (reg / float(spec.torque_limit_reg_max)) * spec.stall_torque_nm


def rad_to_sts3215_ticks(
    q_rad: np.ndarray,
    zero_ticks: np.ndarray | int = 2048,
    spec: STS3215Spec = DEFAULT_SPEC,
) -> np.ndarray:
    """Convert joint radians to 12-bit STS3215 encoder ticks [0..4095]."""
    ticks = np.rint(np.asarray(q_rad, dtype=float) / spec.rad_per_tick + np.asarray(zero_ticks, dtype=float))
    return np.clip(ticks, 0, spec.encoder_ticks - 1).astype(np.int32)


def sts3215_ticks_to_rad(
    ticks: np.ndarray,
    zero_ticks: np.ndarray | int = 2048,
    spec: STS3215Spec = DEFAULT_SPEC,
) -> np.ndarray:
    """Convert 12-bit STS3215 encoder ticks [0..4095] to quantized joint radians."""
    return (np.asarray(ticks, dtype=float) - np.asarray(zero_ticks, dtype=float)) * spec.rad_per_tick


class GravCompTL:
    """Standalone arm-only Gravity-Compensated Servo Torque Limiting controller (< 0.02 ms/step).

    Computes static gravity torque `tau_grav(q)` using an isolated visual-stripped SO-101 MuJoCo
    model (zero scene/object/privileged state) and maps per-joint contact-force margins
    `alpha in [0.05, 1.0]^5` to both physical torque limits (N·m) and STS3215 `0x30` register ticks.
    """

    def __init__(
        self,
        so101_dir: str | Path | None = None,
        tau_margin_max_nm: float | None = None,
        calib_path: str | Path | None = None,
        ee_site_offset_grip: tuple[float, float, float] = (-0.001, 0.0, -0.094),
        gripper_hold_rad: float = -0.10,
        spec: STS3215Spec = DEFAULT_SPEC,
    ):
        model_dir = Path(so101_dir) if so101_dir is not None else DEFAULT_SO101_DIR
        xml_path = model_dir / 'so101.xml'
        assets_dir = model_dir / 'assets'
        if not xml_path.exists():
            raise FileNotFoundError(f'SO-101 Menagerie XML not found at {xml_path}')

        root = ET.fromstring(xml_path.read_text())
        compiler = root.find('compiler')
        assert compiler is not None
        compiler.set('meshdir', str(assets_dir.resolve()))
        opt = root.find('option')
        assert opt is not None
        opt.set('timestep', '0.001')

        gripper_body = root.find(".//body[@name='gripper']")
        assert gripper_body is not None
        ET.SubElement(
            gripper_body, 'site', name='ee_site',
            pos=f'{ee_site_offset_grip[0]:.8g} {ee_site_offset_grip[1]:.8g} {ee_site_offset_grip[2]:.8g}',
            quat='1 0 0 0', size='0.002',
        )
        xml = strip_visual_only_meshes(ET.tostring(root, encoding='unicode'))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
        self.spec = spec
        self.n_joints = 5
        self.tau_margin_max_nm = spec.tau_margin_max_nm if tau_margin_max_nm is None else float(tau_margin_max_nm)
        self.gripper_hold_rad = float(gripper_hold_rad)

        self.grav_scale = np.ones(self.n_joints, dtype=float)
        self.grav_offset_nm = np.zeros(self.n_joints, dtype=float)
        if calib_path is not None:
            self.load_calibration(calib_path)

        self._tau_buf = np.zeros(self.model.nv, dtype=float)
        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)

    @property
    def joint_limits_rad(self) -> np.ndarray:
        """Shape (5, 2) [min, max] joint limits for the 5 SO-101 arm joints."""
        return self.model.jnt_range[:self.n_joints].copy()

    def load_calibration(self, calib_path: str | Path) -> None:
        cal = json.loads(Path(calib_path).read_text())
        self.grav_scale = np.asarray(cal['grav_scale'], dtype=float).reshape(self.n_joints)
        self.grav_offset_nm = np.asarray(cal['grav_offset_nm'], dtype=float).reshape(self.n_joints)

    def _set_q(self, q_rad: np.ndarray) -> None:
        q_in = np.asarray(q_rad, dtype=float).reshape(-1)
        if q_in.size == self.n_joints:
            self.data.qpos[:self.n_joints] = q_in
            self.data.qpos[self.n_joints] = self.gripper_hold_rad
        elif q_in.size >= self.n_joints + 1:
            self.data.qpos[:self.n_joints + 1] = q_in[:self.n_joints + 1]
        else:
            raise ValueError(f'Expected 5 or 6 joint angles, got shape {q_in.shape}')
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0

    def compute_gravity_torque(self, q_rad: np.ndarray) -> np.ndarray:
        """Static gravity torque tau_grav(q) in N·m via MuJoCo `mj_rne` (< 0.02 ms)."""
        self._set_q(q_rad)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self._tau_buf[:] = 0.0
        mujoco.mj_rne(self.model, self.data, 0, self._tau_buf)
        return self.grav_scale * self._tau_buf[:self.n_joints] + self.grav_offset_nm

    def compute_fk_jacobian(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Nominal arm-only FK pose (pos, R) and 3x5 translational/rotational Jacobians."""
        self._set_q(q_rad)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        pos = self.data.site_xpos[self.ee_site_id].copy()
        R = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.ee_site_id)
        return pos, R, self._jacp[:, :self.n_joints].copy(), self._jacr[:, :self.n_joints].copy()

    def compute_limits(self, q_rad: np.ndarray, alpha: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(tau_lim_nm, reg_0x30_ticks, tau_grav_nm)` for per-joint `alpha in [0.05, 1.0]^5`."""
        tau_g = self.compute_gravity_torque(q_rad)
        a = expand_alpha(alpha, n_joints=self.n_joints, spec=self.spec)
        lim_nm = np.clip(
            np.abs(tau_g) + a * self.tau_margin_max_nm,
            self.spec.min_torque_lim_nm,
            self.spec.stall_torque_nm,
        )
        reg_ticks = nm_to_sts3215_ticks(lim_nm, spec=self.spec)
        return lim_nm, reg_ticks, tau_g


def compute_tactile_snr(
    err_stiff_rad: np.ndarray,
    err_gravcomp_rad: np.ndarray,
    threshold_counts: float = 2.0,
    spec: STS3215Spec = DEFAULT_SPEC,
) -> dict[str, float]:
    """Quantify Tactile-Amplifier gain (median L2 counts and fraction above encoder floor)."""
    e_s = np.linalg.norm(np.atleast_2d(err_stiff_rad), axis=-1) / spec.rad_per_tick
    e_g = np.linalg.norm(np.atleast_2d(err_gravcomp_rad), axis=-1) / spec.rad_per_tick
    med_s, med_g = float(np.median(e_s)), float(np.median(e_g))
    return {
        'stiff_median_counts': med_s,
        'gravcomp_median_counts': med_g,
        'amplification_ratio': med_g / max(med_s, 1e-9),
        'stiff_frac_above_thresh': float(np.mean(e_s > threshold_counts)),
        'gravcomp_frac_above_thresh': float(np.mean(e_g > threshold_counts)),
    }


def compute_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Exact rank-based AUROC (Mann-Whitney U) with zero scipy dependency."""
    pos = np.asarray(pos_scores, dtype=float).reshape(-1)
    neg = np.asarray(neg_scores, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    sorted_s = scores[order]
    n = scores.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = float(np.sum(ranks[:pos.size])) - 0.5 * pos.size * (pos.size + 1)
    return u / float(pos.size * neg.size)
