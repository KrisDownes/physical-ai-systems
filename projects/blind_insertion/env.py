"""Blind proprioceptive peg-in-hole insertion environment on the MuJoCo SO-101 arm.

Reuses ONLY the Menagerie `robotstudio_so101` model (`so101.xml` + STL assets) already
present in this repo, with zero bottle-task code reused.

Physics & hardware fidelity:
- 1 kHz MuJoCo physics (timestep = 0.001 s), 50 Hz policy (20 substeps, DT = 0.02 s).
- Headless-first (render=False by default; optional EGL rendering for video/debug).
- Exact convex peg + socket prisms from `mjfit` (bolted Regime A or unbolted Regime B).
- 12-bit STS3215 magnetic encoder quantization (2*pi/4096 rad) on observed q and q_dot.
- Stateful STS3215 gear backlash / deadband proxy (0.003-0.006 rad).
- Noisy + 1-step (20 ms) lagged motor-current proxy (data.actuator_force, +-8% mult + 0.02 Nm add).
- Dynamic per-joint torque-limit scaling (torque_scale in [0.15, 1.0]) directly clamping
  model.actuator_forcerange[:5] to match the Feetech STS3215 Torque Limit register.
- Strict actor obs vs training-only info['privileged'] separation.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .geometry import (
    AssemblyGeometrySpec,
    extract_mjfit_elements,
    hole_outer_radius_m,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SO101_MODEL_DIR = REPO_ROOT / 'projects' / 'active_robot_adaptation' / '.so101-model'
SO101_XML_PATH = SO101_MODEL_DIR / 'so101.xml'
SO101_ASSETS_DIR = SO101_MODEL_DIR / 'assets'

PHYSICS_TIMESTEP = 0.001          # 1 kHz MuJoCo physics
SUBSTEPS = 20                     # 20 substeps per policy step
DT = PHYSICS_TIMESTEP * SUBSTEPS  # 0.02 s -> 50 Hz policy
ENCODER_QUANT_RAD = 2.0 * math.pi / 4096.0  # 12-bit magnetic encoder (~0.001534 rad)
STS3215_MAX_TORQUE_NM = 2.94
TORQUE_SCALE_MIN = 0.15
TORQUE_SCALE_MAX = 1.0

# Peg mounting offset inside SO-101 `gripper` frame:
# fixed_jaw_sph_tip1 sits at z=-0.101 (capsule at z=-0.103).
# Placing the 36 mm peg center at z=-0.107 puts peg top at -0.089 (14 mm tucked inside
# closed jaws) and peg bottom tip at -0.125 (22 mm below jaw tips -> 2 mm clearance above
# the 20 mm deep socket plate even at full bottom-out!).
PEG_MOUNT_BASE_GRIP_FRAME = np.array([0.005, 0.0, -0.107], dtype=float)
GRIPPER_CLOSED_HOLD_RAD = 0.35

# Task-space step limits at 50 Hz (virtual spring target offset from current q_quantized)
MAX_TASK_DELTA_XY_M = 0.004   # 4 mm virtual step -> smooth up to ~160 mm/s without windup
MAX_TASK_DELTA_Z_M = 0.003    # 3 mm virtual step -> smooth compliant preload
MAX_JOINT_DELTA_RAD = 0.06    # safe per-step joint command increment

# Success & failure physics thresholds (zero synthetic shaping rules)
TARGET_SEAT_DEPTH_M = 0.012       # >= 12 mm inside hole
MAX_SOCKET_DISP_MM = 25.0         # unbolted shove failure threshold
MAX_SOCKET_TILT_DEG = 15.0        # unbolted tip failure threshold

ALLOWED_OBS_KEYS = frozenset({
    'q_quantized',
    'q_dot_quantized',
    'ee_pose_fk',
    'ee_twist_fk',
    'prev_action',
    'tracking_error',
    'actuator_force_noisy',
    'torque_scale',
    'time_s',
})

FORBIDDEN_PRIVILEGED_KEYS = frozenset({
    'socket_xy', 'socket_xy_error', 'peg_depth', 'contact_class',
    'peak_contact_force_N', 'socket_displacement_mm', 'socket_tilt_deg',
    'shape', 'clearance_m', 'mu_peg_hole', 'mu_table', 'socket_mass_kg',
    'bolted', 'grasp_offset_xy', 'true_q', 'true_qvel', 'ft_wrench',
})


@dataclass(frozen=True)
class STS3215Config:
    """Feetech STS3215 servo & sensor noise/lag/backlash configuration."""
    encoder_quant_rad: float = ENCODER_QUANT_RAD
    backlash_rad: float = 0.0045     # total mechanical gear lash width (0.003-0.006 rad)
    force_mult_range: float = 0.08   # +-8% multiplicative current-proxy noise
    force_add_std_Nm: float = 0.02   # 0.02 N·m additive Gaussian noise
    force_lag_steps: int = 1         # 1-step = 20 ms lag at 50 Hz


@dataclass(frozen=True)
class InsertionTaskSpec:
    """Full specification for one deterministic peg-in-hole episode."""
    trial_id: str = 'trial-0'
    seed: int = 0
    shape: str = 'round'
    clearance_m: float = 0.0012
    bolted: bool = True
    socket_xy: tuple[float, float] = (0.220, 0.000)
    misalign_xy: tuple[float, float] = (0.010, 0.000)
    start_clearance_z_m: float = 0.002  # peg bottom starts 2 mm above socket top plate
    grasp_offset_xy: tuple[float, float] = (0.0, 0.0)
    mu_peg_hole: float = 0.30
    mu_table: float = 0.45              # unbolted socket-table friction [0.35, 0.55]
    socket_mass_kg: float = 0.250       # unbolted socket mass [0.18, 0.32] kg
    backlash_rad: float = 0.0045        # [0.003, 0.006] rad
    max_steps: int = 250                # 5.0 s at 50 Hz

    def geometry(self) -> AssemblyGeometrySpec:
        return AssemblyGeometrySpec(shape=self.shape, clearance_m=self.clearance_m)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['socket_xy'] = list(self.socket_xy)
        d['misalign_xy'] = list(self.misalign_xy)
        d['grasp_offset_xy'] = list(self.grasp_offset_xy)
        return d


# Cache compiled nominal robot model (for controller-legal FK/IK — zero socket/peg truth!)
_NOMINAL_ARM_MODEL_CACHE: mujoco.MjModel | None = None


def _build_nominal_arm_model() -> mujoco.MjModel:
    """Unperturbed SO-101 arm model with nominal peg_tip site for controller-legal FK/IK."""
    global _NOMINAL_ARM_MODEL_CACHE
    if _NOMINAL_ARM_MODEL_CACHE is not None:
        return _NOMINAL_ARM_MODEL_CACHE

    root = ET.fromstring(SO101_XML_PATH.read_text())
    compiler = root.find('compiler')
    assert compiler is not None
    compiler.set('meshdir', str(SO101_ASSETS_DIR.resolve()))

    opt = root.find('option')
    assert opt is not None
    opt.set('timestep', f'{PHYSICS_TIMESTEP:.6g}')

    # Add nominal peg_tip site at nominal mount position (zero grasp offset, zero socket info).
    gripper_body = root.find(".//body[@name='gripper']")
    assert gripper_body is not None
    tip_pos = PEG_MOUNT_BASE_GRIP_FRAME + np.array([0.0, 0.0, -AssemblyGeometrySpec().peg_length_m / 2.0])
    ET.SubElement(
        gripper_body,
        'site',
        name='nominal_peg_tip',
        pos=f'{tip_pos[0]:.8g} {tip_pos[1]:.8g} {tip_pos[2]:.8g}',
        quat='1 0 0 0',
        size='0.002',
    )
    xml = ET.tostring(root, encoding='unicode')
    _NOMINAL_ARM_MODEL_CACHE = mujoco.MjModel.from_xml_string(xml)
    return _NOMINAL_ARM_MODEL_CACHE


def _build_scene_xml(task: InsertionTaskSpec) -> str:
    """Compose Menagerie SO-101 + exact mjfit convex peg & socket in memory (zero temp files)."""
    geom_spec = task.geometry()
    meshes, plate_geoms, peg_geoms = extract_mjfit_elements(geom_spec, friction=task.mu_peg_hole)

    root = ET.fromstring(SO101_XML_PATH.read_text())
    root.set('model', f'so101_blind_insertion_{task.shape}')

    # 1 kHz physics & elliptic cone solver for stiff convex peg-in-hole contacts.
    opt = root.find('option')
    assert opt is not None
    opt.set('timestep', f'{PHYSICS_TIMESTEP:.6g}')
    opt.set('integrator', 'implicitfast')
    opt.set('cone', 'elliptic')
    opt.set('iterations', '20')
    opt.set('ls_iterations', '30')
    opt.set('impratio', '10')

    compiler = root.find('compiler')
    assert compiler is not None
    compiler.set('meshdir', str(SO101_ASSETS_DIR.resolve()))

    # Contact default class for mjfit peg & socket geoms (priority=1, table gets priority=2).
    # solref=(0.005, 1.0) at 1 kHz gives 5 substeps/time-constant (critically damped, stiff <0.1mm
    # penetration, zero numerical 1ms impact ringing).
    default_sec = root.find('default')
    assert default_sec is not None
    mjfit_def = ET.SubElement(default_sec, 'default', {'class': 'mjfit'})
    ET.SubElement(
        mjfit_def,
        'geom',
        solref='0.005 1.0',
        solimp='0.92 0.98 0.001 0.5 2.0',
        condim='4',
        priority='1',
        friction=f'{task.mu_peg_hole:.6g} 0.005 0.0005',
    )

    # Append mjfit convex prism meshes into <asset>.
    asset_sec = root.find('asset')
    assert asset_sec is not None
    for mesh_el in meshes:
        asset_sec.append(mesh_el)

    wb = root.find('worldbody')
    assert wb is not None

    # Lighting, overhead camera (for optional EGL rendering), and tabletop plane.
    ET.SubElement(wb, 'light', pos='0.2 0.0 1.0', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(wb, 'camera', name='overhead', pos='0.22 -0.22 0.28', xyaxes='1 0 0 0 0.72 0.69', fovy='42')
    ET.SubElement(
        wb,
        'geom',
        name='table',
        type='plane',
        pos='0 0 0',
        size='1 1 0.01',
        rgba='0.32 0.34 0.38 1',
        condim='4',
        priority='2',
        friction=f'{task.mu_table:.6g} 0.008 0.0005',
        solref='0.005 1.0',
        solimp='0.92 0.98 0.001 0.5 2.0',
    )

    # Socket plate body (bolted or freejoint unbolted on tabletop z=0).
    sx, sy = float(task.socket_xy[0]), float(task.socket_xy[1])
    z_plate_center = geom_spec.bottom_thickness_m + geom_spec.plate_thickness_m / 2.0
    socket_body = ET.SubElement(
        wb,
        'body',
        name='socket_plate',
        childclass='mjfit',
        pos=f'{sx:.8g} {sy:.8g} {z_plate_center:.8g}',
    )
    if not task.bolted:
        ET.SubElement(socket_body, 'freejoint', name='socket_joint')
    ET.SubElement(
        socket_body,
        'site',
        name='socket_top_center',
        pos=f'0 0 {geom_spec.plate_thickness_m / 2.0:.8g}',
        size='0.002',
        rgba='0.2 0.9 0.2 0.5',
    )
    for g in plate_geoms:
        socket_body.append(g)

    # Mount peg body inside SO-101 `gripper` frame with randomized small in-hand grasp offset.
    gripper_body = wb.find(".//body[@name='gripper']")
    assert gripper_body is not None
    mount_pos = PEG_MOUNT_BASE_GRIP_FRAME + np.array(
        [float(task.grasp_offset_xy[0]), float(task.grasp_offset_xy[1]), 0.0],
        dtype=float,
    )
    peg_body = ET.SubElement(
        gripper_body,
        'body',
        name='peg',
        childclass='mjfit',
        pos=f'{mount_pos[0]:.8g} {mount_pos[1]:.8g} {mount_pos[2]:.8g}',
        quat='1 0 0 0',
    )
    ET.SubElement(
        peg_body,
        'site',
        name='peg_tip_true',
        pos=f'0 0 {-geom_spec.peg_length_m / 2.0:.8g}',
        size='0.002',
        rgba='0.95 0.2 0.2 0.8',
    )
    for g in peg_geoms:
        peg_body.append(g)

    # Exclude internal jaw-to-peg contact (peg is held by jaws) while preserving ALL peg-socket,
    # peg-table, socket-table, and robot arm self-collision contacts.
    contact_sec = root.find('contact')
    if contact_sec is None:
        contact_sec = ET.SubElement(root, 'contact')
    for bname in ('gripper', 'moving_jaw_so101_v1', 'camera_mount'):
        ET.SubElement(contact_sec, 'exclude', body1=bname, body2='peg')

    return ET.tostring(root, encoding='unicode')


def solve_vertical_ik(
    nominal_model: mujoco.MjModel,
    nominal_data: mujoco.MjData,
    tip_site_id: int,
    target_xyz: np.ndarray,
    q_init: np.ndarray | None = None,
    n_iters: int = 80,
) -> np.ndarray:
    """Solve 5-DoF IK on nominal arm so peg_tip reaches target_xyz with exact vertical R=I."""
    q = np.array([0.0, -0.08, 0.30, 1.35, 0.0, GRIPPER_CLOSED_HOLD_RAD], dtype=float) if q_init is None else q_init.copy()
    q[5] = GRIPPER_CLOSED_HOLD_RAD
    j_lo = nominal_model.jnt_range[:5, 0] + 0.02
    j_hi = nominal_model.jnt_range[:5, 1] - 0.02
    R_des = np.eye(3, dtype=float)
    jacp = np.zeros((3, nominal_model.nv), dtype=float)
    jacr = np.zeros((3, nominal_model.nv), dtype=float)

    for _ in range(n_iters):
        nominal_data.qpos[:6] = q
        mujoco.mj_forward(nominal_model, nominal_data)
        pos = nominal_data.site_xpos[tip_site_id]
        R = nominal_data.site_xmat[tip_site_id].reshape(3, 3)
        e_pos = target_xyz - pos
        # SO(3) orientation error: 0.5 * sum_i (R[:,i] x R_des[:,i])
        e_ori = 0.5 * (
            np.cross(R[:, 0], R_des[:, 0])
            + np.cross(R[:, 1], R_des[:, 1])
            + np.cross(R[:, 2], R_des[:, 2])
        )
        if np.linalg.norm(e_pos) < 1e-5 and np.linalg.norm(e_ori) < 1e-4:
            break
        mujoco.mj_jacSite(nominal_model, nominal_data, jacp, jacr, tip_site_id)
        J = np.vstack([jacp[:, :5], 0.12 * jacr[:, :5]])
        e = np.concatenate([e_pos, 0.12 * e_ori])
        dq = np.linalg.solve(J.T @ J + 1e-5 * np.eye(5), J.T @ e)
        q[:5] = np.clip(q[:5] + 0.65 * dq, j_lo, j_hi)
    return q


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


class BlindInsertionEnv:
    """Headless-first 50 Hz / 1 kHz blind proprioceptive peg-in-hole SO-101 environment."""

    def __init__(
        self,
        task: InsertionTaskSpec | None = None,
        servo_cfg: STS3215Config | None = None,
        render: bool = False,
        render_width: int = 320,
        render_height: int = 240,
    ):
        self.task = task if task is not None else InsertionTaskSpec()
        self.servo_cfg = servo_cfg if servo_cfg is not None else STS3215Config(backlash_rad=self.task.backlash_rad)
        self.render_enabled = bool(render)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        # Nominal controller-side arm model (strictly zero socket/peg truth).
        self.nominal_model = _build_nominal_arm_model()
        self.nominal_data = mujoco.MjData(self.nominal_model)
        self.nominal_tip_site = mujoco.mj_name2id(self.nominal_model, mujoco.mjtObj.mjOBJ_SITE, 'nominal_peg_tip')

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self._compile_task_model(self.task)

        # Stateful episode variables initialized in reset().
        self.rng = np.random.default_rng(self.task.seed)
        self.step_count = 0
        self._q_cmd = np.zeros(6, dtype=float)
        self._q_backlash = np.zeros(6, dtype=float)
        self._q_quant_prev = np.zeros(6, dtype=float)
        self._torque_scale = np.ones(5, dtype=float)
        self._prev_action = np.zeros(8, dtype=float)
        self._force_lag_buf: deque[np.ndarray] = deque(maxlen=max(self.servo_cfg.force_lag_steps + 1, 1))
        self._last_step_force_mean = np.zeros(6, dtype=float)
        self._last_step_force_abs_max = np.zeros(6, dtype=float)
        self._socket_init_xy = np.zeros(2, dtype=float)

        # Safety & episode telemetry
        self.self_collision_steps = 0
        self.joint_limit_steps = 0
        self.episode_peak_contact_force_N = 0.0
        self.episode_max_socket_disp_mm = 0.0
        self.episode_max_socket_tilt_deg = 0.0
        self._step_peak_contact_N = 0.0
        self._step_peg_socket_contact = False

    def _compile_task_model(self, task: InsertionTaskSpec) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        xml = _build_scene_xml(task)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'peg')
        self.peg_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'peg_tip_true')
        self.socket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'socket_plate')
        self.socket_top_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'socket_top_center')
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        self.moving_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'moving_jaw_so101_v1')

        # Exact unbolted socket mass scaling (preserving uniform-density inertia tensor).
        if not task.bolted:
            cur_mass = float(self.model.body_mass[self.socket_body_id])
            scale = float(task.socket_mass_kg) / max(cur_mass, 1e-9)
            self.model.body_mass[self.socket_body_id] *= scale
            self.model.body_inertia[self.socket_body_id] *= scale
            mujoco.mj_setConst(self.model, self.data)

        # Precompute integer geom ID sets for fast zero-string-lookup 1 kHz auditing.
        self._peg_geoms = frozenset(
            i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) == self.peg_body_id
        )
        self._socket_geoms = frozenset(
            i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) == self.socket_body_id
        )
        self._arm_root_id = int(self.model.body_rootid[self.gripper_body_id])
        self._hole_outer_r = hole_outer_radius_m(task.shape)
        self._geom_spec = task.geometry()
        self._cforce_buf = np.zeros(6, dtype=float)

        if self.render_enabled:
            self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render_rgb(self, camera: str = 'overhead') -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def _quantize_q(self, q: np.ndarray) -> np.ndarray:
        step = self.servo_cfg.encoder_quant_rad
        if step <= 0.0:
            return q.copy()
        return np.round(q / step) * step

    def _apply_backlash(self, q_cmd: np.ndarray) -> np.ndarray:
        """Stateful mechanical gear backlash deadband of total width `backlash_rad`."""
        half = 0.5 * max(float(self.servo_cfg.backlash_rad), 0.0)
        if half <= 0.0:
            self._q_backlash = q_cmd.copy()
            return self._q_backlash.copy()
        diff = q_cmd - self._q_backlash
        self._q_backlash = np.where(
            diff > half,
            q_cmd - half,
            np.where(diff < -half, q_cmd + half, self._q_backlash),
        )
        return self._q_backlash.copy()

    def _apply_torque_limits(self, torque_scale: np.ndarray) -> None:
        ts = np.clip(np.asarray(torque_scale, dtype=float), TORQUE_SCALE_MIN, TORQUE_SCALE_MAX)
        self._torque_scale = ts.copy()
        for j in range(5):
            lim = STS3215_MAX_TORQUE_NM * float(ts[j])
            self.model.actuator_forcerange[j, 0] = -lim
            self.model.actuator_forcerange[j, 1] = lim
        self.model.actuator_forcerange[5, 0] = -STS3215_MAX_TORQUE_NM
        self.model.actuator_forcerange[5, 1] = STS3215_MAX_TORQUE_NM

    def _read_noisy_lagged_force(self, raw_force: np.ndarray) -> np.ndarray:
        f = np.asarray(raw_force, dtype=float).copy()
        if self.servo_cfg.force_mult_range > 0.0:
            f = f * (1.0 + self.rng.uniform(-self.servo_cfg.force_mult_range, self.servo_cfg.force_mult_range, size=6))
        if self.servo_cfg.force_add_std_Nm > 0.0:
            f = f + self.rng.normal(0.0, self.servo_cfg.force_add_std_Nm, size=6)
        self._force_lag_buf.append(f)
        lag = self.servo_cfg.force_lag_steps
        if lag <= 0 or len(self._force_lag_buf) <= lag:
            return self._force_lag_buf[0].copy()
        return self._force_lag_buf[-(lag + 1)].copy()

    def controller_fk(self, q_quant: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Controller-legal nominal FK & Jacobian using ONLY quantized joint angles."""
        self.nominal_data.qpos[:6] = q_quant
        mujoco.mj_forward(self.nominal_model, self.nominal_data)
        pos = self.nominal_data.site_xpos[self.nominal_tip_site].copy()
        R = self.nominal_data.site_xmat[self.nominal_tip_site].reshape(3, 3).copy()
        jacp = np.zeros((3, self.nominal_model.nv), dtype=float)
        jacr = np.zeros((3, self.nominal_model.nv), dtype=float)
        mujoco.mj_jacSite(self.nominal_model, self.nominal_data, jacp, jacr, self.nominal_tip_site)
        return pos, R, jacp[:, :5].copy(), jacr[:, :5].copy()

    def _task_delta_to_joint_delta(self, q_ref: np.ndarray, d_xyz: np.ndarray) -> np.ndarray:
        """Map bounded 3D task-space delta (dx, dy, dz) to 5-DoF joint delta keeping vertical R=I."""
        d_xyz = np.asarray(d_xyz, dtype=float)
        d_xyz = np.array([
            np.clip(d_xyz[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
            np.clip(d_xyz[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
            np.clip(d_xyz[2], -MAX_TASK_DELTA_Z_M, MAX_TASK_DELTA_Z_M),
        ], dtype=float)
        _, R, jp, jr = self.controller_fk(q_ref)
        R_des = np.eye(3, dtype=float)
        e_ori = 0.5 * (
            np.cross(R[:, 0], R_des[:, 0])
            + np.cross(R[:, 1], R_des[:, 1])
            + np.cross(R[:, 2], R_des[:, 2])
        )
        # Keep peg strictly vertical & yaw-aligned to socket while tracking (dx, dy, dz).
        w_ori = 0.12
        J = np.vstack([jp, w_ori * jr])
        e = np.concatenate([d_xyz, w_ori * 0.5 * e_ori])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(5), J.T @ e)
        return np.clip(dq, -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)

    def _audit_substep(self) -> None:
        """Fast integer-ID physics audit at 1 kHz: self-collision, joint limits, contact force."""
        m = self.model
        d = self.data
        q = d.qpos[:6]
        if np.any(q < m.jnt_range[:6, 0] - 0.002) or np.any(q > m.jnt_range[:6, 1] + 0.002):
            self.joint_limit_steps += 1

        ncon = int(d.ncon)
        if ncon <= 0:
            return
        g1s = d.contact.geom1[:ncon]
        g2s = d.contact.geom2[:ncon]
        dists = d.contact.dist[:ncon]
        self_col = False
        for i in range(ncon):
            g1 = int(g1s[i])
            g2 = int(g2s[i])
            # Peg-socket contact force measurement (evaluator/privileged only).
            if (g1 in self._peg_geoms and g2 in self._socket_geoms) or (g2 in self._peg_geoms and g1 in self._socket_geoms):
                self._step_peg_socket_contact = True
                mujoco.mj_contactForce(m, d, i, self._cforce_buf)
                fmag = float(np.linalg.norm(self._cforce_buf[:3]))
                if fmag > self._step_peak_contact_N:
                    self._step_peak_contact_N = fmag
            # Arm self-collision check (excluding expected gripper-jaw pair).
            if dists[i] < -1e-5:
                b1 = int(m.geom_bodyid[g1])
                b2 = int(m.geom_bodyid[g2])
                if b1 > 0 and b2 > 0 and int(m.body_rootid[b1]) == self._arm_root_id and int(m.body_rootid[b2]) == self._arm_root_id:
                    pair = {b1, b2}
                    if pair != {self.gripper_body_id, self.moving_jaw_body_id}:
                        self_col = True
        if self_col:
            self.self_collision_steps += 1

    def _compute_privileged_info(self) -> dict:
        peg_tip = self.data.site_xpos[self.peg_tip_site_id].copy()
        sock_top = self.data.site_xpos[self.socket_top_site_id].copy()
        xy_err = peg_tip[:2] - sock_top[:2]
        radial_err = float(np.linalg.norm(xy_err))
        peg_depth = float(sock_top[2] - peg_tip[2])

        sock_xy_now = self.data.xpos[self.socket_body_id, :2]
        sock_disp_mm = 0.0 if self.task.bolted else float(np.linalg.norm(sock_xy_now - self._socket_init_xy) * 1000.0)
        r_zz = float(self.data.xmat[self.socket_body_id].reshape(3, 3)[2, 2])
        sock_tilt_deg = 0.0 if self.task.bolted else float(np.degrees(np.arccos(np.clip(r_zz, -1.0, 1.0))))

        if sock_disp_mm > self.episode_max_socket_disp_mm:
            self.episode_max_socket_disp_mm = sock_disp_mm
        if sock_tilt_deg > self.episode_max_socket_tilt_deg:
            self.episode_max_socket_tilt_deg = sock_tilt_deg
        if self._step_peak_contact_N > self.episode_peak_contact_force_N:
            self.episode_peak_contact_force_N = self._step_peak_contact_N

        # Clean physical contact classification (privileged training-only label).
        if peg_depth >= TARGET_SEAT_DEPTH_M and radial_err < self._hole_outer_r:
            c_class = 'seated'
        elif peg_depth >= 0.002 and radial_err < self._hole_outer_r:
            c_class = 'entering'
        elif self._step_peg_socket_contact:
            if radial_err <= self._hole_outer_r + 0.003:
                c_class = 'rim_contact'
            else:
                c_class = 'plate_contact'
        else:
            c_class = 'free'

        return {
            'socket_xy_error': xy_err.copy(),
            'peg_depth': peg_depth,
            'contact_class': c_class,
            'peak_contact_force_N': float(self._step_peak_contact_N),
            'socket_displacement_mm': sock_disp_mm,
            'socket_tilt_deg': sock_tilt_deg,
            'shape': self.task.shape,
            'clearance_m': self.task.clearance_m,
            'mu_peg_hole': self.task.mu_peg_hole,
            'mu_table': self.task.mu_table,
            'socket_mass_kg': self.task.socket_mass_kg,
            'bolted': self.task.bolted,
        }

    def _build_obs(self, q_quant: np.ndarray, q_dot_quant: np.ndarray, f_noisy: np.ndarray) -> dict:
        pos, R, jp, jr = self.controller_fk(q_quant)
        quat = _mat_to_quat_wxyz(R)
        v_lin = jp @ q_dot_quant[:5]
        v_ang = jr @ q_dot_quant[:5]
        obs = {
            'q_quantized': q_quant.copy(),
            'q_dot_quantized': q_dot_quant.copy(),
            'ee_pose_fk': np.concatenate([pos, quat]),
            'ee_twist_fk': np.concatenate([v_lin, v_ang]),
            'prev_action': self._prev_action.copy(),
            'tracking_error': (self._q_cmd - q_quant).copy(),
            'actuator_force_noisy': f_noisy.copy(),
            'torque_scale': self._torque_scale.copy(),
            'time_s': float(self.step_count * DT),
        }
        return obs

    def reset(
        self,
        task: InsertionTaskSpec | None = None,
        servo_cfg: STS3215Config | None = None,
    ) -> tuple[dict, dict]:
        """Reset episode cleanly with peg positioned above socket + lateral misalignment."""
        if task is not None:
            self.task = task
            if servo_cfg is None:
                self.servo_cfg = STS3215Config(backlash_rad=task.backlash_rad)
            else:
                self.servo_cfg = servo_cfg
            self._compile_task_model(task)
        elif servo_cfg is not None:
            self.servo_cfg = servo_cfg

        self.rng = np.random.default_rng(self.task.seed + 104729)
        self.step_count = 0
        self.self_collision_steps = 0
        self.joint_limit_steps = 0
        self.episode_peak_contact_force_N = 0.0
        self.episode_max_socket_disp_mm = 0.0
        self.episode_max_socket_tilt_deg = 0.0
        self._step_peak_contact_N = 0.0
        self._step_peg_socket_contact = False

        mujoco.mj_resetData(self.model, self.data)
        self._apply_torque_limits(np.ones(5, dtype=float))

        # Place peg tip at socket_xy + misalign_xy, start_clearance_z_m above plate top.
        start_xyz = np.array([
            self.task.socket_xy[0] + self.task.misalign_xy[0],
            self.task.socket_xy[1] + self.task.misalign_xy[1],
            self._geom_spec.plate_top_z_m + self.task.start_clearance_z_m,
        ], dtype=float)
        q0 = solve_vertical_ik(self.nominal_model, self.nominal_data, self.nominal_tip_site, start_xyz)
        self.data.qpos[:6] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:6] = q0
        if not self.task.bolted:
            # Freejoint qpos starts at index 6: [x, y, z, qw, qx, qy, qz]
            z_plate = self._geom_spec.bottom_thickness_m + self._geom_spec.plate_thickness_m / 2.0
            self.data.qpos[6:9] = [self.task.socket_xy[0], self.task.socket_xy[1], z_plate]
            self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

        # Brief 60 ms settle so gravity torque and socket table contact equilibrate cleanly.
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._socket_init_xy = self.data.xpos[self.socket_body_id, :2].copy()
        self._q_cmd = q0.copy()
        self._q_backlash = q0.copy()
        q_quant = self._quantize_q(self.data.qpos[:6])
        self._q_quant_prev = q_quant.copy()
        self._prev_action = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
        self._last_step_force_mean = self.data.actuator_force[:6].copy()
        self._last_step_force_abs_max = np.abs(self._last_step_force_mean)

        self._force_lag_buf = deque(maxlen=max(self.servo_cfg.force_lag_steps + 1, 1))
        f_noisy = self._read_noisy_lagged_force(self._last_step_force_mean)

        obs = self._build_obs(q_quant, np.zeros(6, dtype=float), f_noisy)
        priv = self._compute_privileged_info()
        info = {
            'privileged': priv,
            'success': False,
            'failure_mode': None,
            'self_collision_steps': 0,
            'joint_limit_steps': 0,
            'episode_peak_contact_force_N': 0.0,
            'episode_max_socket_disp_mm': 0.0,
            'episode_max_socket_tilt_deg': 0.0,
            'step_force_abs_max_Nm': self._last_step_force_abs_max.copy(),
        }
        return obs, info

    def _parse_action(self, action, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse task-space or joint-space action + optional torque_scale into (dq_5, ts_5, prev_act_8)."""
        if isinstance(action, dict):
            ts_raw = action.get('torque_scale', np.ones(5, dtype=float))
            ts = np.full(5, float(ts_raw), dtype=float) if np.ndim(ts_raw) == 0 else np.asarray(ts_raw, dtype=float)
            ts = np.clip(ts, TORQUE_SCALE_MIN, TORQUE_SCALE_MAX)
            if 'delta_q' in action:
                dq = np.clip(np.asarray(action['delta_q'], dtype=float), -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
                prev_act = np.concatenate([np.zeros(3), ts])
            else:
                d_xyz = np.asarray(action['delta_xyz'], dtype=float)
                dq = self._task_delta_to_joint_delta(q_ref, d_xyz)
                prev_act = np.concatenate([
                    [np.clip(d_xyz[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
                     np.clip(d_xyz[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
                     np.clip(d_xyz[2], -MAX_TASK_DELTA_Z_M, MAX_TASK_DELTA_Z_M)],
                    ts,
                ])
            return dq, ts, prev_act

        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size == 3:
            d_xyz = arr
            ts = np.ones(5, dtype=float)
        elif arr.size == 4:
            d_xyz = arr[:3]
            ts = np.full(5, float(np.clip(arr[3], TORQUE_SCALE_MIN, TORQUE_SCALE_MAX)), dtype=float)
        elif arr.size == 8:
            d_xyz = arr[:3]
            ts = np.clip(arr[3:8], TORQUE_SCALE_MIN, TORQUE_SCALE_MAX)
        else:
            raise ValueError(f'Expected action len 3, 4, or 8 (or dict), got shape {arr.shape}')
        dq = self._task_delta_to_joint_delta(q_ref, d_xyz)
        prev_act = np.concatenate([
            [np.clip(d_xyz[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
             np.clip(d_xyz[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
             np.clip(d_xyz[2], -MAX_TASK_DELTA_Z_M, MAX_TASK_DELTA_Z_M)],
            ts,
        ])
        return dq, ts, prev_act

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """Advance 1 policy step (20 ms = 20 x 1 kHz substeps) with active STS3215 torque limits."""
        # Anchor delta command to current quantized joint position (standard compliant virtual-target
        # impedance interface: prevents open-loop integrator windup into solid contact or joint limits).
        q_anchor = self._q_quant_prev.copy()
        dq, ts, prev_act = self._parse_action(action, q_anchor)
        self._apply_torque_limits(ts)

        j_lo = self.model.jnt_range[:5, 0] + 0.02
        j_hi = self.model.jnt_range[:5, 1] - 0.02
        self._q_cmd[:5] = np.clip(q_anchor[:5] + dq, j_lo, j_hi)
        self._q_cmd[5] = GRIPPER_CLOSED_HOLD_RAD
        q_eff = self._apply_backlash(self._q_cmd)
        self.data.ctrl[:6] = q_eff

        self._step_peak_contact_N = 0.0
        self._step_peg_socket_contact = False
        f_accum = np.zeros(6, dtype=float)
        f_abs_max = np.zeros(6, dtype=float)

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            f_now = self.data.actuator_force[:6]
            f_accum += f_now
            np.maximum(f_abs_max, np.abs(f_now), out=f_abs_max)
            self._audit_substep()

        self.step_count += 1
        self._prev_action = prev_act
        self._last_step_force_mean = f_accum / SUBSTEPS
        self._last_step_force_abs_max = f_abs_max

        q_quant = self._quantize_q(self.data.qpos[:6])
        q_dot_quant = (q_quant - self._q_quant_prev) / DT
        self._q_quant_prev = q_quant.copy()
        f_noisy = self._read_noisy_lagged_force(self._last_step_force_mean)

        obs = self._build_obs(q_quant, q_dot_quant, f_noisy)
        priv = self._compute_privileged_info()

        # Clean termination logic (zero synthetic shortcuts).
        shoved = (not self.task.bolted) and (priv['socket_displacement_mm'] > MAX_SOCKET_DISP_MM)
        tipped = (not self.task.bolted) and (priv['socket_tilt_deg'] > MAX_SOCKET_TILT_DEG)
        safety_bad = (self.self_collision_steps > 0) or (self.joint_limit_steps > 0)
        seated = (priv['contact_class'] == 'seated') and (not shoved) and (not tipped) and (not safety_bad)

        terminated = bool(seated or shoved or tipped or safety_bad)
        truncated = bool((not terminated) and (self.step_count >= self.task.max_steps))

        if seated:
            failure_mode = 'success'
            reward = 1.0
        elif safety_bad:
            failure_mode = 'safety_violation'
            reward = -1.0
        elif shoved:
            failure_mode = 'socket_shoved'
            reward = -0.5
        elif tipped:
            failure_mode = 'socket_tipped'
            reward = -0.5
        elif truncated:
            if priv['peg_depth'] >= 0.002:
                failure_mode = 'stalled_in_hole'
            elif priv['contact_class'] == 'rim_contact':
                failure_mode = 'jammed_on_rim'
            else:
                failure_mode = 'jammed_on_plate'
            reward = 0.0
        else:
            failure_mode = None
            reward = 0.0

        info = {
            'privileged': priv,
            'success': bool(seated),
            'failure_mode': failure_mode,
            'self_collision_steps': self.self_collision_steps,
            'joint_limit_steps': self.joint_limit_steps,
            'episode_peak_contact_force_N': self.episode_peak_contact_force_N,
            'episode_max_socket_disp_mm': self.episode_max_socket_disp_mm,
            'episode_max_socket_tilt_deg': self.episode_max_socket_tilt_deg,
            'step_force_abs_max_Nm': f_abs_max.copy(),
            'time_s': float(self.step_count * DT),
        }
        return obs, reward, terminated, truncated, info
