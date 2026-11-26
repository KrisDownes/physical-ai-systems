"""Blind proprioceptive unbolted sticky/latched drawer opening environment on the MuJoCo SO-101 arm.

Physics & hardware fidelity (matched 1-to-1 with real Feetech STS3215 hardware):
- 1 kHz MuJoCo physics (`timestep = 0.001 s`), 50 Hz policy (`20 substeps`, `DT = 0.02 s`).
- 12-bit STS3215 encoder quantization (`2*pi/4096 rad`), `0.0045 rad` stateful gear backlash
  hysteresis, and `GravComp-TL` actuator mode (`clip(|tau_grav_i(q_quant)| + alpha_i*1.50, 0.05, 2.94)`).
- Real SO-101 native jaw grasp (`fixed_jaw` + `moving_jaw_so101_v1`) closed cleanly around the
  horizontal drawer handle at step 0 (NO welded constraints — realistic jaw friction so excessive
  off-axis binding torque can pry/slip the handle out of the jaw!).
- Articulated cabinet + prismatic drawer (`travel in [0, 90] mm`, target `>= 70 mm`) with:
  1. Unknown rail yaw `theta_rail in [-18, +18]` deg (`+-22` deg in Split OOD).
  2. Unknown sliding friction `f_rail in [1.5, 5.5] N` + real Coulomb off-axis rail-binding
     friction `mu_bind * |F_lateral|` + smooth physical spring-detent catch bump at unknown
     travel `d_detent in [10, 40] mm` requiring `F_detent in [3.0, 9.0] N` extra axial pull.
  3. Unbolted tabletop cabinet (`mass 280-750 g`, `mu_table 0.35-0.65`), failing if cabinet
     slides `> 25 mm` or tilts `> 10 deg`.
- Strict relative-kinematics actor observation boundary (ZERO absolute `ee_pose_fk`, ZERO privileged keys).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .gravcomp import (
    GRAVCOMP_MIN_LIM_NM,
    PHYSICS_TIMESTEP,
    SO101_ASSETS_DIR,
    SO101_XML_PATH,
    STS3215_MAX_TORQUE_NM,
    compute_gravcomp_torque_limits_nm,
    strip_visual_only_meshes,
)

SUBSTEPS = 20
DT = PHYSICS_TIMESTEP * SUBSTEPS  # 0.02 s -> 50 Hz policy
ENCODER_QUANT_RAD = 2.0 * math.pi / 4096.0
TORQUE_SCALE_MIN = 0.10
TORQUE_SCALE_MAX = 1.00

# Native SO-101 jaw grasp center in `gripper` frame (between fixed_jaw x=-9mm and moving_jaw x=+2..+10mm)
JAW_GRASP_CENTER_GRIP_FRAME = np.array([-0.001, 0.0, -0.094], dtype=float)
GRIPPER_CLAMP_CTRL_RAD = -0.11   # firm realistic grasp (~15-20 N clamp; slips under excessive off-axis prying)
GRIPPER_START_Q_RAD = -0.05      # initial closed position snugly contacting the 9mm bar at step 0

MAX_TASK_DELTA_XY_M = 0.0070
MAX_TASK_DELTA_Z_M = 0.0070
MAX_JOINT_DELTA_RAD = 0.100

# Cabinet & prismatic drawer physical dimensions (all meters)
CABINET_HALF_X = 0.058           # 116 mm cabinet shell depth
CABINET_HALF_Y = 0.058           # 116 mm cabinet width
BASE_HANDLE_Z_M = 0.045          # nominal handle center height (Split OOD shifts +-15 mm -> 30 / 60 mm!)
DRAWER_MAX_TRAVEL_M = 0.090      # 90 mm physical rail stroke
HANDLE_X_LOCAL = -0.078          # bar center 20 mm in front of drawer front face (-0.058)
HANDLE_HALF_X = 0.0045           # 9 mm dia smooth horizontal bar (fits cleanly inside closed SO-101 jaws!)
HANDLE_HALF_Y = 0.036            # 72 mm wide horizontal grab bar (clears jaws cleanly across +-22 deg yaw!)
HANDLE_HALF_Z = 0.0045           # 9 mm bar height

# Success & physical failure thresholds
TARGET_TRAVEL_MM = 70.0          # >= 70 mm drawer opening
MAX_CABINET_DISP_MM = 25.0       # unbolted cabinet slide failure threshold
MAX_CABINET_TILT_DEG = 10.0      # unbolted cabinet tilt failure threshold
HANDLE_SLIP_DIST_M = 0.028       # jaw center > 28 mm from handle bar -> slipped out of jaws

# Strictly controller-legal actor observation keys (ZERO absolute ee_pose_fk, ZERO privileged info!)
ALLOWED_OBS_KEYS = frozenset({
    'q_quantized',
    'q_dot_quantized',
    'ee_rel_start',
    'ee_twist_fk',
    'tracking_error',
    'prev_action',
})

FORBIDDEN_PRIVILEGED_KEYS = frozenset({
    'ee_pose_fk', 'ee_pos_fk', 'ee_world_xyz',
    'drawer_travel_mm', 'rail_yaw_rad', 'cabinet_disp_mm', 'cabinet_tilt_deg',
    'peak_contact_force_N', 'contact_class', 'handle_pos_world', 'detent_travel_mm',
    'detent_force_N', 'f_rail_N', 'cabinet_mass_kg', 'mu_table', 'bolted',
    'true_q', 'true_qvel', 'ft_wrench',
})


@dataclass(frozen=True)
class STS3215Config:
    encoder_quant_rad: float = ENCODER_QUANT_RAD
    backlash_rad: float = 0.0045


@dataclass(frozen=True)
class DrawerTaskSpec:
    """Full specification for one deterministic blind drawer-opening episode."""
    trial_id: str = 'drawer-0'
    seed: int = 0
    bolted: bool = True
    cabinet_xy: tuple[float, float] = (0.336, 0.000)  # closed handle starts at x ~ 0.258 m
    height_shift_m: float = 0.0          # 0.0 nominal (Split A/B); +-0.015 m in Split OOD!
    rail_yaw_deg: float = 0.0            # unknown rail yaw [-18, +18] deg (+-22 deg OOD)
    f_rail_N: float = 2.5                # base sliding rail friction [1.5, 5.5] N
    rail_damping: float = 12.0           # viscous slide damping (N*s/m)
    rail_bind_mu: float = 0.50           # Coulomb off-axis rail-binding coefficient (mu_bind * |F_lat|)
    detent_travel_mm: float = 22.0       # hidden sticky detent location [10, 40] mm
    detent_force_N: float = 5.5          # hidden sticky detent extra axial pull [3.0, 9.0] N
    detent_width_mm: float = 10.0        # smooth physical detent bump half-width
    mu_jaw_handle: float = 0.85          # realistic rubberized SO-101 jaw-on-handle friction
    mu_table: float = 0.52               # unbolted cabinet-table friction [0.40, 0.65] (0.35 OOD)
    cabinet_mass_kg: float = 0.550       # unbolted cabinet mass [0.40, 0.75] kg (0.28-0.35 OOD)
    backlash_rad: float = 0.0045         # STS3215 mechanical gear backlash hysteresis
    max_steps: int = 300                 # 6.0 s at 50 Hz

    def to_dict(self) -> dict:
        d = asdict(self)
        d['cabinet_xy'] = list(self.cabinet_xy)
        return d


_NOMINAL_ARM_MODEL_CACHE: mujoco.MjModel | None = None


def _build_nominal_arm_model() -> mujoco.MjModel:
    """Unperturbed lean SO-101 arm model with nominal jaw_grasp_center site (zero cabinet/drawer info)."""
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

    gripper_body = root.find(".//body[@name='gripper']")
    assert gripper_body is not None
    ET.SubElement(
        gripper_body, 'site', name='nominal_jaw_center',
        pos=f'{JAW_GRASP_CENTER_GRIP_FRAME[0]:.8g} {JAW_GRASP_CENTER_GRIP_FRAME[1]:.8g} {JAW_GRASP_CENTER_GRIP_FRAME[2]:.8g}',
        quat='1 0 0 0', size='0.002',
    )
    xml = strip_visual_only_meshes(ET.tostring(root, encoding='unicode'))
    _NOMINAL_ARM_MODEL_CACHE = mujoco.MjModel.from_xml_string(xml)
    return _NOMINAL_ARM_MODEL_CACHE


def _yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    h = 0.5 * float(yaw_rad)
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)], dtype=float)


def solve_grasp_ik(
    nominal_model: mujoco.MjModel,
    nominal_data: mujoco.MjData,
    site_id: int,
    target_xyz: np.ndarray,
    yaw_rad: float = 0.0,
    q_init: np.ndarray | None = None,
    n_iters: int = 90,
) -> np.ndarray:
    """Solve 5-DoF IK on nominal SO-101 arm placing jaw center at `target_xyz` aligned vertically with `yaw_rad`."""
    q = np.array([0.0, -0.08, 0.30, 1.35, 0.0, GRIPPER_START_Q_RAD], dtype=float) if q_init is None else q_init.copy()
    q[5] = GRIPPER_START_Q_RAD
    j_lo = nominal_model.jnt_range[:5, 0] + 0.02
    j_hi = nominal_model.jnt_range[:5, 1] - 0.02
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R_des = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    jacp = np.zeros((3, nominal_model.nv), dtype=float)
    jacr = np.zeros((3, nominal_model.nv), dtype=float)

    for _ in range(n_iters):
        nominal_data.qpos[:6] = q
        mujoco.mj_forward(nominal_model, nominal_data)
        pos = nominal_data.site_xpos[site_id]
        R = nominal_data.site_xmat[site_id].reshape(3, 3)
        e_pos = target_xyz - pos
        e_ori = 0.5 * (
            np.cross(R[:, 0], R_des[:, 0])
            + np.cross(R[:, 1], R_des[:, 1])
            + np.cross(R[:, 2], R_des[:, 2])
        )
        if np.linalg.norm(e_pos) < 1e-5 and np.linalg.norm(e_ori) < 1e-4:
            break
        mujoco.mj_jacSite(nominal_model, nominal_data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :5], 0.14 * jacr[:, :5]])
        e = np.concatenate([e_pos, 0.14 * e_ori])
        dq = np.linalg.solve(J.T @ J + 1e-5 * np.eye(5), J.T @ e)
        q[:5] = np.clip(q[:5] + 0.65 * dq, j_lo, j_hi)
    return q


def _build_drawer_scene_xml(task: DrawerTaskSpec, lean_headless: bool = True) -> str:
    """Build MuJoCo XML for Menagerie SO-101 + tabletop cabinet + prismatic drawer + native jaw grasp."""
    root = ET.fromstring(SO101_XML_PATH.read_text())
    root.set('model', f'so101_blind_drawer_{"bolted" if task.bolted else "unbolted"}')

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

    # Slightly increase gripper servo closing torque headroom & friction class for realistic jaw grasp
    default_sec = root.find('default')
    assert default_sec is not None
    drw_def = ET.SubElement(default_sec, 'default', {'class': 'drawer_part'})
    ET.SubElement(
        drw_def, 'geom',
        solref='0.005 1.0', solimp='0.92 0.98 0.001 0.5 2.0',
        condim='4', priority='1',
        friction=f'{task.mu_jaw_handle:.6g} 0.010 0.001',
    )

    wb = root.find('worldbody')
    assert wb is not None
    ET.SubElement(wb, 'light', pos='0.2 0.0 1.0', dir='0 0 -1', diffuse='0.85 0.85 0.85')
    ET.SubElement(wb, 'camera', name='overhead', pos='0.22 -0.26 0.28', xyaxes='1 0 0 0 0.72 0.69', fovy='44')

    # Table surface shifts cleanly with cabinet height_shift_m (+-15 mm in Split OOD)
    table_z = float(task.height_shift_m)
    ET.SubElement(
        wb, 'geom', name='table', type='plane', pos=f'0 0 {table_z:.8g}', size='1 1 0.01',
        rgba='0.32 0.34 0.38 1', condim='4', priority='2',
        friction=f'{task.mu_table:.6g} 0.008 0.0005',
        solref='0.005 1.0', solimp='0.92 0.98 0.001 0.5 2.0',
    )

    # Cabinet shell resting flat on table_z with unknown yaw
    cx, cy = float(task.cabinet_xy[0]), float(task.cabinet_xy[1])
    cq = _yaw_quat_wxyz(math.radians(task.rail_yaw_deg))
    cab_body = ET.SubElement(
        wb, 'body', name='cabinet', childclass='drawer_part',
        pos=f'{cx:.8g} {cy:.8g} {table_z:.8g}',
        quat=f'{cq[0]:.8g} {cq[1]:.8g} {cq[2]:.8g} {cq[3]:.8g}',
    )
    if not task.bolted:
        ET.SubElement(cab_body, 'freejoint', name='cabinet_joint')

    # Distributed cabinet mass (`cabinet_mass_kg`), low COM and wide flat base footprint (prevents spurious tipping)
    m_c = float(task.cabinet_mass_kg)
    ix = m_c * (0.14**2 + 0.055**2) / 12.0
    iy = m_c * (0.16**2 + 0.055**2) / 12.0
    iz = m_c * (0.16**2 + 0.14**2) / 12.0
    ET.SubElement(
        cab_body, 'inertial', pos='0.005 0 0.022', mass=f'{m_c:.6g}',
        diaginertia=f'{ix:.8g} {iy:.8g} {iz:.8g}',
    )
    # Cabinet walls: weighted desktop baseplate extending under the drawer front (prevents forward nose-tipping)
    cab_walls = [
        ('cab_bottom', '-0.025 0 0.003', '0.090 0.068 0.003', '0.45 0.48 0.54 1'),
        ('cab_left', '0 -0.054 0.032', f'{CABINET_HALF_X} 0.004 0.026', '0.48 0.52 0.58 1'),
        ('cab_right', '0 0.054 0.032', f'{CABINET_HALF_X} 0.004 0.026', '0.48 0.52 0.58 1'),
        ('cab_back', '0.054 0 0.032', '0.004 0.050 0.026', '0.42 0.45 0.50 1'),
        ('cab_top', '0.008 0 0.060', '0.050 0.058 0.0025', '0.50 0.54 0.60 1'),
    ]
    for gname, gpos, gsize, grgba in cab_walls:
        ET.SubElement(cab_body, 'geom', name=gname, type='box', pos=gpos, size=gsize, rgba=grgba, mass='0')

    # Sliding prismatic drawer (`travel in [0, 90] mm` along -x_local toward robot)
    drw_body = ET.SubElement(cab_body, 'body', name='drawer', childclass='drawer_part', pos='0 0 0')
    ET.SubElement(
        drw_body, 'joint', name='drawer_slide', type='slide', axis='-1 0 0',
        limited='true', range=f'0 {DRAWER_MAX_TRAVEL_M:.6g}',
        frictionloss=f'{task.f_rail_N:.6g}',
        damping=f'{task.rail_damping:.6g}',
        armature='0.01',
    )
    ET.SubElement(
        drw_body, 'inertial', pos='-0.012 0 0.030', mass='0.10',
        diaginertia='0.00014 0.00018 0.00022',
    )
    # Drawer box + horizontal bar handle at z = BASE_HANDLE_Z_M (45 mm above cabinet bottom), x = -78 mm
    hz = BASE_HANDLE_Z_M
    drw_geoms = [
        ('drw_bottom', '-0.004 0 0.010', '0.050 0.047 0.002', '0.78 0.66 0.46 1'),
        ('drw_front', '-0.056 0 0.032', '0.003 0.048 0.020', '0.84 0.70 0.48 1'),
        ('drw_left', '-0.004 -0.046 0.032', '0.048 0.002 0.018', '0.76 0.64 0.44 1'),
        ('drw_right', '-0.004 0.046 0.032', '0.048 0.002 0.018', '0.76 0.64 0.44 1'),
        # Side standoffs (y = +-36 mm) + lower handle shelf directly under jaw tips so downward stabilization
        # transfers cleanly into the cabinet at ALL heights (30 / 45 / 60 mm) without jaw slip!
        ('handle_post_l', f'-0.068 -{HANDLE_HALF_Y:.6g} {hz:.6g}', '0.011 0.0025 0.0035', '0.25 0.27 0.30 1'),
        ('handle_post_r', f'-0.068 {HANDLE_HALF_Y:.6g} {hz:.6g}', '0.011 0.0025 0.0035', '0.25 0.27 0.30 1'),
    ]
    for gname, gpos, gsize, grgba in drw_geoms:
        ET.SubElement(drw_body, 'geom', name=gname, type='box', pos=gpos, size=gsize, rgba=grgba, mass='0')
    # Low-friction vertical stabilization ledge 1.5 mm below jaw tips (zero spawn overlap; transfers -Fz cleanly)
    ET.SubElement(
        drw_body, 'geom', name='handle_ledge', type='box',
        pos=f'{HANDLE_X_LOCAL:.6g} 0 {hz - 0.0135:.6g}', size=f'0.010 {HANDLE_HALF_Y:.6g} 0.0025',
        rgba='0.28 0.30 0.34 1', mass='0', priority='2', friction='0.05 0.002 0.0005',
    )
    ET.SubElement(
        drw_body, 'geom', name='handle_bar', type='capsule',
        fromto=f'{HANDLE_X_LOCAL:.6g} {-HANDLE_HALF_Y:.6g} {hz:.6g} {HANDLE_X_LOCAL:.6g} {HANDLE_HALF_Y:.6g} {hz:.6g}',
        size=f'{HANDLE_HALF_X:.6g}', rgba='0.20 0.22 0.25 1', mass='0',
    )

    ET.SubElement(
        drw_body, 'site', name='handle_center',
        pos=f'{HANDLE_X_LOCAL:.6g} 0 {hz:.6g}', size='0.002', rgba='0.2 0.9 0.2 0.6',
    )

    # Add jaw grasp center site directly onto SO-101 `gripper` body (no synthetic attachments!)
    gripper_body = wb.find(".//body[@name='gripper']")
    assert gripper_body is not None
    ET.SubElement(
        gripper_body, 'site', name='jaw_grasp_center',
        pos=f'{JAW_GRASP_CENTER_GRIP_FRAME[0]:.8g} {JAW_GRASP_CENTER_GRIP_FRAME[1]:.8g} {JAW_GRASP_CENTER_GRIP_FRAME[2]:.8g}',
        quat='1 0 0 0', size='0.002', rgba='0.95 0.2 0.2 0.8',
    )

    contact_sec = root.find('contact')
    if contact_sec is None:
        contact_sec = ET.SubElement(root, 'contact')
    # Drawer slides cleanly on prismatic joint inside cabinet shell (friction + binding handled on drawer_slide)
    ET.SubElement(contact_sec, 'exclude', body1='cabinet', body2='drawer')

    xml_str = ET.tostring(root, encoding='unicode')
    return strip_visual_only_meshes(xml_str) if lean_headless else xml_str


class BlindDrawerEnv:
    """Headless-first 50 Hz / 1 kHz blind proprioceptive drawer-opening SO-101 environment."""

    def __init__(
        self,
        task: DrawerTaskSpec | None = None,
        servo_cfg: STS3215Config | None = None,
        render: bool = False,
        render_width: int = 320,
        render_height: int = 240,
    ):
        self.task = task if task is not None else DrawerTaskSpec()
        self.servo_cfg = servo_cfg if servo_cfg is not None else STS3215Config(backlash_rad=self.task.backlash_rad)
        self.render_enabled = bool(render)
        self.render_width = int(render_width)
        self.render_height = int(render_height)

        self.nominal_model = _build_nominal_arm_model()
        self.nominal_data = mujoco.MjData(self.nominal_model)
        self.nominal_site_id = mujoco.mj_name2id(self.nominal_model, mujoco.mjtObj.mjOBJ_SITE, 'nominal_jaw_center')

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self._compile_task_model(self.task)

        self.rng = np.random.default_rng(self.task.seed)
        self.step_count = 0
        self._q_cmd = np.zeros(6, dtype=float)
        self._q_backlash = np.zeros(6, dtype=float)
        self._q_quant_prev = np.zeros(6, dtype=float)
        self._start_ee_fk = np.zeros(3, dtype=float)
        self._torque_scale = np.ones(5, dtype=float)
        self._pending_gravcomp_lim_nm: np.ndarray | None = None
        self._prev_action = np.zeros(8, dtype=float)
        self._cab_init_xy = np.zeros(2, dtype=float)
        self._cab_init_yaw = 0.0

        self.self_collision_steps = 0
        self.joint_limit_steps = 0
        self.episode_peak_contact_force_N = 0.0
        self.episode_max_cab_disp_mm = 0.0
        self.episode_max_cab_tilt_deg = 0.0
        self._step_peak_contact_N = 0.0
        self._step_lat_force_N = 0.0
        self._step_jaw_handle_contact = False

    def _compile_task_model(self, task: DrawerTaskSpec) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        xml = _build_drawer_scene_xml(task, lean_headless=(not self.render_enabled))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.cab_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cabinet')
        self.drw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'drawer')
        self.handle_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'handle_center')
        self.jaw_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'jaw_grasp_center')
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        self.moving_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'moving_jaw_so101_v1')

        drw_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'drawer_slide')
        self.drawer_qpos_adr = int(self.model.jnt_qposadr[drw_jnt_id])
        self.drawer_dof_adr = int(self.model.jnt_dofadr[drw_jnt_id])

        self._jaw_geoms = frozenset(
            i for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in (self.gripper_body_id, self.moving_jaw_body_id)
        )
        self._handle_geoms = frozenset(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, nm)
            for nm in ('handle_bar', 'handle_post_l', 'handle_post_r')
        )
        self._handle_bar_gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'handle_bar'))
        self._drw_cab_geoms = frozenset(
            i for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in (self.cab_body_id, self.drw_body_id)
        )
        self._arm_root_id = int(self.model.body_rootid[self.gripper_body_id])
        self._cforce_buf = np.zeros(6, dtype=float)

        if self.render_enabled:
            self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _quantize_q(self, q: np.ndarray) -> np.ndarray:
        step = self.servo_cfg.encoder_quant_rad
        return q.copy() if step <= 0.0 else np.round(q / step) * step

    def _apply_backlash(self, q_cmd: np.ndarray) -> np.ndarray:
        half = 0.5 * max(float(self.servo_cfg.backlash_rad), 0.0)
        if half <= 0.0:
            self._q_backlash = q_cmd.copy()
            return self._q_backlash.copy()
        diff = q_cmd - self._q_backlash
        self._q_backlash = np.where(
            diff > half, q_cmd - half, np.where(diff < -half, q_cmd + half, self._q_backlash)
        )
        return self._q_backlash.copy()

    def _apply_torque_limits(self, torque_scale: np.ndarray) -> None:
        if self._pending_gravcomp_lim_nm is not None:
            lims = np.clip(self._pending_gravcomp_lim_nm, GRAVCOMP_MIN_LIM_NM, STS3215_MAX_TORQUE_NM)
            self._pending_gravcomp_lim_nm = None
            self._torque_scale = lims / STS3215_MAX_TORQUE_NM
            for j in range(5):
                lim = float(lims[j])
                self.model.actuator_forcerange[j, 0] = -lim
                self.model.actuator_forcerange[j, 1] = lim
        else:
            ts = np.clip(np.asarray(torque_scale, dtype=float), TORQUE_SCALE_MIN, TORQUE_SCALE_MAX)
            self._torque_scale = ts.copy()
            for j in range(5):
                lim = STS3215_MAX_TORQUE_NM * float(ts[j])
                self.model.actuator_forcerange[j, 0] = -lim
                self.model.actuator_forcerange[j, 1] = lim
        # Gripper servo keeps full nominal holding torque (2.94 Nm)
        self.model.actuator_forcerange[5, 0] = -STS3215_MAX_TORQUE_NM
        self.model.actuator_forcerange[5, 1] = STS3215_MAX_TORQUE_NM

    def controller_fk(self, q_quant: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Controller-legal nominal FK & Jacobian using ONLY quantized joint angles."""
        self.nominal_data.qpos[:6] = q_quant
        mujoco.mj_forward(self.nominal_model, self.nominal_data)
        pos = self.nominal_data.site_xpos[self.nominal_site_id].copy()
        R = self.nominal_data.site_xmat[self.nominal_site_id].reshape(3, 3).copy()
        jacp = np.zeros((3, self.nominal_model.nv), dtype=float)
        jacr = np.zeros((3, self.nominal_model.nv), dtype=float)
        mujoco.mj_jacSite(self.nominal_model, self.nominal_data, jacp, jacr, self.nominal_site_id)
        return pos, R, jacp[:, :5].copy(), jacr[:, :5].copy()

    def _task_delta_to_joint_delta(self, q_ref: np.ndarray, d_xyz: np.ndarray) -> np.ndarray:
        """Map bounded 3D task-space delta (dx, dy, dz) to 5-DoF joint delta (vertical jaw orientation)."""
        d_xyz = np.array([
            np.clip(d_xyz[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
            np.clip(d_xyz[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
            np.clip(d_xyz[2], -MAX_TASK_DELTA_Z_M, MAX_TASK_DELTA_Z_M),
        ], dtype=float)
        _, R, jp, jr = self.controller_fk(q_ref)
        # Keep jaw pitch/roll vertical (R[:,2] -> [0,0,1]) while letting wrist yaw yield gently to rail!
        e_tilt = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
        w_ori = 0.14
        J = np.vstack([jp, w_ori * jr[:2]])
        e = np.concatenate([d_xyz, w_ori * 0.5 * e_tilt[:2]])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(5), J.T @ e)
        return np.clip(dq, -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)

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
        arr = np.asarray(action, dtype=float).reshape(-1)
        if arr.size == 4:
            d_xyz = arr[:3]
            ts = np.full(5, float(arr[3]), dtype=float)
        elif arr.size == 8:
            d_xyz = arr[:3]
            ts = arr[3:8].copy()
        else:
            raise ValueError(f'Expected 4-dim, 8-dim, or gravcomp dict action; got {action!r}')
        dq = self._task_delta_to_joint_delta(q_ref, d_xyz)
        ts = np.clip(ts, TORQUE_SCALE_MIN, TORQUE_SCALE_MAX)
        prev_act = np.concatenate([
            [np.clip(d_xyz[0], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
             np.clip(d_xyz[1], -MAX_TASK_DELTA_XY_M, MAX_TASK_DELTA_XY_M),
             np.clip(d_xyz[2], -MAX_TASK_DELTA_Z_M, MAX_TASK_DELTA_Z_M)],
            ts,
        ])
        return dq, ts, prev_act

    def _audit_substep_and_measure_lateral_load(self) -> float:
        """1 kHz safety audit + measure real lateral contact force perpendicular to drawer rail."""
        m = self.model
        d = self.data
        q = d.qpos[:6]
        if np.any(q < m.jnt_range[:6, 0] - 0.002) or np.any(q > m.jnt_range[:6, 1] + 0.002):
            self.joint_limit_steps += 1

        ncon = int(d.ncon)
        if ncon > 0:
            g1s = d.contact.geom1[:ncon]
            g2s = d.contact.geom2[:ncon]
            dists = d.contact.dist[:ncon]
            self_col = False
            for i in range(ncon):
                g1, g2 = int(g1s[i]), int(g2s[i])
                if (g1 in self._jaw_geoms and g2 in self._drw_cab_geoms) or (g2 in self._jaw_geoms and g1 in self._drw_cab_geoms):
                    mujoco.mj_contactForce(m, d, i, self._cforce_buf)
                    fmag = float(np.linalg.norm(self._cforce_buf[:3]))
                    if fmag > self._step_peak_contact_N:
                        self._step_peak_contact_N = fmag
                    if g1 in self._handle_geoms or g2 in self._handle_geoms:
                        self._step_jaw_handle_contact = True
                if dists[i] < -1e-5:
                    b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
                    if b1 > 0 and b2 > 0 and int(m.body_rootid[b1]) == self._arm_root_id and int(m.body_rootid[b2]) == self._arm_root_id:
                        if {b1, b2} != {self.gripper_body_id, self.moving_jaw_body_id}:
                            self_col = True
            if self_col:
                self.self_collision_steps += 1

        # Exact lateral rail-binding load transmitted across prismatic drawer joint via mj_rnePostConstraint
        mujoco.mj_rnePostConstraint(m, d)
        R_cab = d.xmat[self.cab_body_id].reshape(3, 3)
        f_int_world = d.cfrc_int[self.drw_body_id, 3:6]
        return abs(float(np.dot(f_int_world, R_cab[:, 1])))

    def _apply_detent_and_rail_binding(self, f_lat_abs: float) -> None:
        """Apply physical smooth detent catch bump + real Coulomb off-axis rail-binding friction."""
        q_mm = float(self.data.qpos[self.drawer_qpos_adr]) * 1000.0
        v_drw = float(self.data.qvel[self.drawer_dof_adr])

        # 1. Smooth physical spring-detent barrier centered at `detent_travel_mm` (width `detent_width_mm`)
        f_det = 0.0
        if self.task.detent_force_N > 0.0:
            u = (q_mm - float(self.task.detent_travel_mm)) / max(float(self.task.detent_width_mm), 1.0)
            if abs(u) < 1.0:
                f_det = 0.5 * float(self.task.detent_force_N) * (1.0 + math.cos(math.pi * u))

        # 2. Real Coulomb off-axis rail-binding resistance: mu_bind * max(|F_rail_lat| - 4.5 N, 0)
        f_bind = float(self.task.rail_bind_mu) * max(float(f_lat_abs) - 4.5, 0.0)
        v_dir = math.tanh(v_drw / 0.004)
        self.data.qfrc_applied[self.drawer_dof_adr] = -(f_det + f_bind * max(v_dir, 0.0))

    def _compute_privileged_info(self) -> dict:
        jaw_pos = self.data.site_xpos[self.jaw_site_id].copy()
        handle_pos = self.data.site_xpos[self.handle_site_id].copy()
        R_cab = self.data.xmat[self.cab_body_id].reshape(3, 3)
        rail_yaw = math.atan2(float(R_cab[1, 0]), float(R_cab[0, 0]))
        travel_mm = float(self.data.qpos[self.drawer_qpos_adr]) * 1000.0

        if self.task.bolted:
            cab_disp_mm = 0.0
            cab_tilt_deg = 0.0
        else:
            cab_disp_mm = float(np.linalg.norm(self.data.xpos[self.cab_body_id, :2] - self._cab_init_xy) * 1000.0)
            cab_tilt_deg = float(math.degrees(math.acos(np.clip(float(R_cab[2, 2]), -1.0, 1.0))))

        self.episode_max_cab_disp_mm = max(self.episode_max_cab_disp_mm, cab_disp_mm)
        self.episode_max_cab_tilt_deg = max(self.episode_max_cab_tilt_deg, cab_tilt_deg)
        self.episode_peak_contact_force_N = max(self.episode_peak_contact_force_N, self._step_peak_contact_N)

        handle_dist = float(np.linalg.norm(jaw_pos - handle_pos))
        slipped = (handle_dist > HANDLE_SLIP_DIST_M) and (not self._step_jaw_handle_contact)
        in_detent = abs(travel_mm - float(self.task.detent_travel_mm)) <= float(self.task.detent_width_mm)

        if travel_mm >= TARGET_TRAVEL_MM and not slipped:
            c_class = 'opened'
        elif slipped:
            c_class = 'handle_slip'
        elif self._step_lat_force_N >= 7.0:
            c_class = 'off_axis_bind'
        elif in_detent and self.task.detent_force_N > 0.0:
            c_class = 'detent_catch'
        else:
            c_class = 'free_slide'

        return {
            'drawer_travel_mm': travel_mm,
            'rail_yaw_rad': rail_yaw,
            'cabinet_disp_mm': cab_disp_mm,
            'cabinet_tilt_deg': cab_tilt_deg,
            'peak_contact_force_N': float(self._step_peak_contact_N),
            'lateral_bind_force_N': float(self._step_lat_force_N),
            'contact_class': c_class,
            'handle_pos_world': handle_pos,
            'jaw_pos_world': jaw_pos,
            'handle_dist_mm': handle_dist * 1000.0,
            'handle_slipped': bool(slipped),
            'detent_travel_mm': float(self.task.detent_travel_mm),
            'detent_force_N': float(self.task.detent_force_N),
            'f_rail_N': float(self.task.f_rail_N),
            'cabinet_mass_kg': float(self.task.cabinet_mass_kg),
            'mu_table': float(self.task.mu_table),
            'bolted': bool(self.task.bolted),
        }

    def _build_obs(self, q_quant: np.ndarray, q_dot_quant: np.ndarray) -> dict:
        pos, _, jp, jr = self.controller_fk(q_quant)
        v_lin = jp @ q_dot_quant[:5]
        v_ang = jr @ q_dot_quant[:5]
        return {
            'q_quantized': q_quant.copy(),
            'q_dot_quantized': q_dot_quant.copy(),
            'ee_rel_start': (pos - self._start_ee_fk).copy(),
            'ee_twist_fk': np.concatenate([v_lin, v_ang]),
            'tracking_error': (self._q_cmd - q_quant).copy(),
            'prev_action': self._prev_action.copy(),
        }

    def reset(self, task: DrawerTaskSpec | None = None, servo_cfg: STS3215Config | None = None) -> tuple[dict, dict]:
        if task is not None:
            self.task = task
            self.servo_cfg = servo_cfg if servo_cfg is not None else STS3215Config(backlash_rad=task.backlash_rad)
            self._compile_task_model(task)
        elif servo_cfg is not None:
            self.servo_cfg = servo_cfg

        self.rng = np.random.default_rng(self.task.seed + 104729)
        self.step_count = 0
        self.self_collision_steps = 0
        self.joint_limit_steps = 0
        self.episode_peak_contact_force_N = 0.0
        self.episode_max_cab_disp_mm = 0.0
        self.episode_max_cab_tilt_deg = 0.0
        self._step_peak_contact_N = 0.0
        self._step_lat_force_N = 0.0
        self._step_jaw_handle_contact = True
        self._pending_gravcomp_lim_nm = None

        mujoco.mj_resetData(self.model, self.data)
        self._apply_torque_limits(np.ones(5, dtype=float))

        yaw = math.radians(self.task.rail_yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        cx, cy = float(self.task.cabinet_xy[0]), float(self.task.cabinet_xy[1])
        table_z = float(self.task.height_shift_m)
        handle_world = np.array([
            cx + c * HANDLE_X_LOCAL,
            cy + s * HANDLE_X_LOCAL,
            table_z + BASE_HANDLE_Z_M,
        ], dtype=float)

        # Solve IK placing native SO-101 jaw center cleanly around handle bar (aligned with handle yaw)
        q0 = solve_grasp_ik(self.nominal_model, self.nominal_data, self.nominal_site_id, handle_world, yaw_rad=yaw)
        self.data.qpos[:6] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:5] = q0[:5]
        self.data.ctrl[5] = GRIPPER_CLAMP_CTRL_RAD  # close moving_jaw firmly around handle bar!
        if not self.task.bolted:
            self.data.qpos[6:9] = [cx, cy, table_z]
            self.data.qpos[9:13] = _yaw_quat_wxyz(yaw)
        self.data.qpos[self.drawer_qpos_adr] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Brief 80 ms settle so moving_jaw clamps cleanly onto handle bar and cabinet settles on tabletop
        for _ in range(80):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._cab_init_xy = self.data.xpos[self.cab_body_id, :2].copy()
        R0 = self.data.xmat[self.cab_body_id].reshape(3, 3)
        self._cab_init_yaw = math.atan2(float(R0[1, 0]), float(R0[0, 0]))
        self._q_cmd = self.data.qpos[:6].copy()
        self._q_cmd[5] = GRIPPER_CLAMP_CTRL_RAD
        self._q_backlash = self._q_cmd.copy()
        q_quant = self._quantize_q(self.data.qpos[:6])
        self._q_quant_prev = q_quant.copy()
        pos0, _, _, _ = self.controller_fk(q_quant)
        self._start_ee_fk = pos0.copy()
        self._prev_action = np.zeros(8, dtype=float)

        obs = self._build_obs(q_quant, np.zeros(6, dtype=float))
        priv = self._compute_privileged_info()
        info = {
            'trial_id': self.task.trial_id,
            'step': 0,
            'time_s': 0.0,
            'success': False,
            'failure_mode': 'in_progress',
            'self_collision_steps': 0,
            'joint_limit_steps': 0,
            'episode_peak_contact_force_N': 0.0,
            'episode_max_cab_disp_mm': 0.0,
            'episode_max_cab_tilt_deg': 0.0,
            'privileged': priv,
        }
        return obs, info

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        q_quant_cur = self._q_quant_prev
        dq, ts, prev_act = self._parse_action(action, q_quant_cur)
        self._apply_torque_limits(ts)

        j_lo = self.model.jnt_range[:5, 0] + 0.015
        j_hi = self.model.jnt_range[:5, 1] - 0.015
        self._q_cmd[:5] = np.clip(q_quant_cur[:5] + dq, j_lo, j_hi)
        self._q_cmd[5] = GRIPPER_CLAMP_CTRL_RAD
        q_eff = self._apply_backlash(self._q_cmd)
        self.data.ctrl[:5] = q_eff[:5]
        self.data.ctrl[5] = GRIPPER_CLAMP_CTRL_RAD

        self._step_peak_contact_N = 0.0
        self._step_lat_force_N = 0.0
        self._step_jaw_handle_contact = False
        lat_acc = 0.0

        for _ in range(SUBSTEPS):
            f_lat = self._audit_substep_and_measure_lateral_load()
            lat_acc = max(lat_acc, f_lat)
            self._apply_detent_and_rail_binding(f_lat)
            mujoco.mj_step(self.model, self.data)

        self._step_lat_force_N = lat_acc
        self.step_count += 1
        self._prev_action = prev_act
        q_quant = self._quantize_q(self.data.qpos[:6])
        q_dot_quant = (q_quant - self._q_quant_prev) / DT
        self._q_quant_prev = q_quant.copy()

        obs = self._build_obs(q_quant, q_dot_quant)
        priv = self._compute_privileged_info()

        success = False
        terminated = False
        truncated = False
        failure_mode = 'in_progress'

        if not self.task.bolted:
            if priv['cabinet_tilt_deg'] > MAX_CABINET_TILT_DEG:
                terminated = True
                failure_mode = 'cabinet_tipped'
            elif priv['cabinet_disp_mm'] > MAX_CABINET_DISP_MM:
                terminated = True
                failure_mode = 'cabinet_shoved'

        if not terminated and priv['handle_slipped']:
            terminated = True
            failure_mode = 'handle_slip'

        if not terminated and priv['drawer_travel_mm'] >= TARGET_TRAVEL_MM:
            success = True
            terminated = True
            failure_mode = 'success'
        elif not terminated and self.step_count >= self.task.max_steps:
            truncated = True
            if priv['contact_class'] == 'off_axis_bind':
                failure_mode = 'off_axis_bind_stall'
            elif priv['contact_class'] == 'detent_catch':
                failure_mode = 'detent_stall'
            else:
                failure_mode = 'timeout_partial'

        reward = 1.0 if success else 0.0
        info = {
            'trial_id': self.task.trial_id,
            'step': self.step_count,
            'time_s': float(self.step_count * DT),
            'success': success,
            'failure_mode': failure_mode,
            'self_collision_steps': self.self_collision_steps,
            'joint_limit_steps': self.joint_limit_steps,
            'episode_peak_contact_force_N': self.episode_peak_contact_force_N,
            'episode_max_cab_disp_mm': self.episode_max_cab_disp_mm,
            'episode_max_cab_tilt_deg': self.episode_max_cab_tilt_deg,
            'privileged': priv,
        }
        return obs, reward, terminated, truncated, info
