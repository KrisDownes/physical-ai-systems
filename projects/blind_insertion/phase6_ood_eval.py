"""Phase 6 Zero-Shot OOD Stress Suite Evaluator (ZERO retraining, ZERO policy edits).

Evaluates the frozen Phase 5 checkpoints (`R1_v2_Res_PosOnly` and `R2_v2_Res_GravComp_TL`,
all 3 seeds `[101, 202, 303]`) alongside `BlindSpiral_Stiff_1.0`, `BlindSpiral_GravComp_0.35`,
and `Privileged_Oracle` across 5 physically distinct Out-of-Distribution stress suites
(20 deterministic trials each = 100 OOD trials/model, disjoint seeds, zero edits to Phase 0-5):

  1. `OOD_Height_Tilt`      : Fixture height shifted +-12 mm AND tilted 3-5 deg
  2. `OOD_Spring_Mount`     : Compliant 3-axis spring-damper mount (k_z=800 N/m, k_xy=600 N/m)
  3. `OOD_NonConvex_Asym`   : Non-convex L-shape & keyed/notched holes via mjfit (1.0 mm)
  4. `OOD_Grasp_Error`      : Unobserved +-3.5 mm grasp offset AND 4-6 deg in-hand peg tilt
  5. `OOD_UltraTight_0.30mm`: Sub-encoder-count 0.30 mm clearance across seen + unseen shapes
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
import mjfit
import mujoco
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import projects.blind_insertion.env as _env_mod
from projects.blind_insertion.benchmarks import (
    TRAIN_SEED_OFFSET,
    build_regime_a_bolted,
    build_regime_b_unbolted,
    build_split_c_tight_0_5mm,
)
from projects.blind_insertion.controllers import BlindSpiralController, PrivilegedOracleController
from projects.blind_insertion.env import (
    GRIPPER_CLOSED_HOLD_RAD,
    PEG_MOUNT_BASE_GRIP_FRAME,
    PHYSICS_TIMESTEP,
    SO101_ASSETS_DIR,
    SO101_XML_PATH,
    TARGET_SEAT_DEPTH_M,
    InsertionTaskSpec,
    STS3215Config,
    solve_vertical_ik,
)
from projects.blind_insertion.geometry import ALL_SHAPES, AssemblyGeometrySpec, SEEN_SHAPES, UNSEEN_SHAPES, hole_outer_radius_m
from projects.blind_insertion.phase1_policy import CONTACT_CLASS_TO_IDX
from projects.blind_insertion.phase2_train import P2_PLATE_HARD_HALF_M, build_slippery_unbolted_ood_suite
from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
from projects.blind_insertion.phase4_policy import BASE_SPIRAL_ALPHA, Phase4ResidualBuilder
from projects.blind_insertion.phase4_train import PausableBaseSpiralController, aggregate_seeds_p4, summarize_suite_p4
from projects.blind_insertion.phase5_policy import Phase5ResidualPPO
from projects.blind_insertion.phase5_train import _init_worker_single_thread, _malloc_trim, get_rss_mb

OOD_SUITE_ORDER: tuple[str, ...] = (
    'OOD_Height_Tilt',
    'OOD_Spring_Mount',
    'OOD_NonConvex_Asym',
    'OOD_Grasp_Error',
    'OOD_UltraTight_0.30mm',
)

# Non-convex asymmetric polygons (18 mm bounding box, 100% drop-test verified at 1.0 mm clearance)
NONCONVEX_POLYGONS: dict[str, np.ndarray] = {
    'l_shape': np.array([
        [-0.009, -0.009], [0.009, -0.009], [0.009, -0.001],
        [-0.001, -0.001], [-0.001, 0.009], [-0.009, 0.009],
    ], dtype=float),
    'keyed_notch': np.array([
        [-0.009, -0.009], [0.009, -0.009], [0.009, -0.003],
        [0.004, -0.003], [0.004, 0.003], [0.009, 0.003],
        [0.009, 0.009], [-0.009, 0.009],
    ], dtype=float),
}


def _axis_angle_quat_wxyz(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(ax))
    if n < 1e-12 or abs(angle_rad) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    ax = ax / n
    h = 0.5 * float(angle_rad)
    return np.array([math.cos(h), ax[0] * math.sin(h), ax[1] * math.sin(h), ax[2] * math.sin(h)], dtype=float)


def _hole_outer_radius_ood(shape: str) -> float:
    if shape in NONCONVEX_POLYGONS:
        poly = NONCONVEX_POLYGONS[shape]
        poly = poly - poly.mean(axis=0)
        return float(np.max(np.linalg.norm(poly, axis=1)))
    return hole_outer_radius_m(shape)


@dataclass(frozen=True)
class OODTaskSpec:
    """Wrapper around InsertionTaskSpec adding Phase 6 OOD physical perturbations."""
    base: InsertionTaskSpec
    suite: str
    socket_dz_m: float = 0.0          # OOD_Height_Tilt: +-0.012 m
    socket_tilt_deg: float = 0.0      # OOD_Height_Tilt: 3..5 deg
    socket_tilt_phi: float = 0.0      # horizontal tilt axis angle (rad)
    spring_mount: bool = False        # OOD_Spring_Mount: 3-axis spring-damper
    spring_k_xy: float = 600.0        # N/m
    spring_k_z: float = 800.0         # N/m
    spring_damping: float = 16.0      # N*s/m
    grasp_tilt_deg: float = 0.0       # OOD_Grasp_Error: 4..6 deg in-hand peg tilt
    grasp_tilt_phi: float = 0.0       # horizontal tilt axis angle (rad)


def _extract_nonconvex_mjfit_elements(shape: str, clearance_m: float, friction: float) -> tuple[list[ET.Element], list[ET.Element], list[ET.Element]]:
    spec = AssemblyGeometrySpec(shape='round', clearance_m=clearance_m)
    xml = mjfit.peg(
        hole=NONCONVEX_POLYGONS[shape],
        clearance=clearance_m,
        peg_length=spec.peg_length_m,
        plate_size=spec.plate_size_m,
        plate_thickness=spec.plate_thickness_m,
        bottom_thickness=spec.bottom_thickness_m,
        peg_mass=spec.peg_mass_kg,
        plate_density=1500.0,
        spawn_height=0.001,
        center=True,
        include_peg=True,
        free_plate=False,
        friction=(float(friction), 0.005, 0.0005),
        condim=4,
        timestep=PHYSICS_TIMESTEP,
        standalone=False,
        floor=False,
    )
    root = ET.fromstring(xml)
    asset = root.find('asset')
    wb = root.find('worldbody')
    assert asset is not None and wb is not None
    meshes = list(asset.findall('mesh'))
    plate_body = wb.find("./body[@name='plate']")
    peg_body = wb.find("./body[@name='peg']")
    assert plate_body is not None and peg_body is not None
    return meshes, list(plate_body.findall('geom')), list(peg_body.findall('geom'))


def _build_ood_scene_xml(ood: OODTaskSpec) -> str:
    """Build MuJoCo XML for any Phase 6 OOD task without touching `env.py` or `geometry.py`."""
    task = ood.base
    geom_spec = AssemblyGeometrySpec(shape='round', clearance_m=task.clearance_m)
    if task.shape in NONCONVEX_POLYGONS:
        meshes, plate_geoms, peg_geoms = _extract_nonconvex_mjfit_elements(task.shape, task.clearance_m, task.mu_peg_hole)
    else:
        from projects.blind_insertion.geometry import extract_mjfit_elements
        meshes, plate_geoms, peg_geoms = extract_mjfit_elements(task.geometry(), friction=task.mu_peg_hole)

    root = ET.fromstring(SO101_XML_PATH.read_text())
    root.set('model', f'so101_ood_{ood.suite}_{task.shape}')

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

    asset_sec = root.find('asset')
    assert asset_sec is not None
    for mesh_el in meshes:
        asset_sec.append(mesh_el)

    wb = root.find('worldbody')
    assert wb is not None
    ET.SubElement(wb, 'light', pos='0.2 0.0 1.0', dir='0 0 -1', diffuse='0.8 0.8 0.8')
    ET.SubElement(wb, 'camera', name='overhead', pos='0.22 -0.22 0.28', xyaxes='1 0 0 0 0.72 0.69', fovy='42')

    # Table plane shifted with fixture height (or lowered beneath spring mount)
    table_z = ood.socket_dz_m - (0.040 if ood.spring_mount else 0.0)
    if abs(ood.socket_tilt_deg) > 1e-6:
        table_z -= 0.015  # keep flat table clear of tilted bolted socket corners
    ET.SubElement(
        wb,
        'geom',
        name='table',
        type='plane',
        pos=f'0 0 {table_z:.8g}',
        size='1 1 0.01',
        rgba='0.32 0.34 0.38 1',
        condim='4',
        priority='2',
        friction=f'{task.mu_table:.6g} 0.008 0.0005',
        solref='0.005 1.0',
        solimp='0.92 0.98 0.001 0.5 2.0',
    )

    sx, sy = float(task.socket_xy[0]), float(task.socket_xy[1])
    z_plate_center = geom_spec.bottom_thickness_m + geom_spec.plate_thickness_m / 2.0 + ood.socket_dz_m
    sock_quat = _axis_angle_quat_wxyz(
        np.array([math.cos(ood.socket_tilt_phi), math.sin(ood.socket_tilt_phi), 0.0]),
        math.radians(ood.socket_tilt_deg),
    )
    socket_body = ET.SubElement(
        wb,
        'body',
        name='socket_plate',
        childclass='mjfit',
        pos=f'{sx:.8g} {sy:.8g} {z_plate_center:.8g}',
        quat=f'{sock_quat[0]:.8g} {sock_quat[1]:.8g} {sock_quat[2]:.8g} {sock_quat[3]:.8g}',
    )
    if ood.spring_mount:
        # Compliant 3-axis spring-damper mount (k_xy=600 N/m, k_z=800 N/m)
        ET.SubElement(
            socket_body, 'joint', name='spring_x', type='slide', axis='1 0 0',
            stiffness=f'{ood.spring_k_xy:.6g}', damping=f'{ood.spring_damping:.6g}',
            limited='true', range='-0.025 0.025',
        )
        ET.SubElement(
            socket_body, 'joint', name='spring_y', type='slide', axis='0 1 0',
            stiffness=f'{ood.spring_k_xy:.6g}', damping=f'{ood.spring_damping:.6g}',
            limited='true', range='-0.025 0.025',
        )
        ET.SubElement(
            socket_body, 'joint', name='spring_z', type='slide', axis='0 0 1',
            stiffness=f'{ood.spring_k_z:.6g}', damping=f'{ood.spring_damping:.6g}',
            limited='true', range='-0.025 0.025',
        )
    elif not task.bolted:
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

    gripper_body = wb.find(".//body[@name='gripper']")
    assert gripper_body is not None
    mount_pos = PEG_MOUNT_BASE_GRIP_FRAME + np.array(
        [float(task.grasp_offset_xy[0]), float(task.grasp_offset_xy[1]), 0.0],
        dtype=float,
    )
    peg_quat = _axis_angle_quat_wxyz(
        np.array([math.cos(ood.grasp_tilt_phi), math.sin(ood.grasp_tilt_phi), 0.0]),
        math.radians(ood.grasp_tilt_deg),
    )
    peg_body = ET.SubElement(
        gripper_body,
        'body',
        name='peg',
        childclass='mjfit',
        pos=f'{mount_pos[0]:.8g} {mount_pos[1]:.8g} {mount_pos[2]:.8g}',
        quat=f'{peg_quat[0]:.8g} {peg_quat[1]:.8g} {peg_quat[2]:.8g} {peg_quat[3]:.8g}',
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

    contact_sec = root.find('contact')
    if contact_sec is None:
        contact_sec = ET.SubElement(root, 'contact')
    for bname in ('gripper', 'moving_jaw_so101_v1', 'camera_mount'):
        ET.SubElement(contact_sec, 'exclude', body1=bname, body2='peg')

    return ET.tostring(root, encoding='unicode')


class Phase6OODEnv(GravCompBlindInsertionEnv):
    """Unmodified physics + GravComp-TL evaluator supporting Phase 6 OOD stress perturbations."""

    def __init__(self, ood_task: OODTaskSpec, render: bool = False):
        self.ood_task = ood_task
        super().__init__(task=ood_task.base, render=render)

    def _compile_task_model(self, task: InsertionTaskSpec) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        xml = _build_ood_scene_xml(self.ood_task)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'peg')
        self.peg_tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'peg_tip_true')
        self.socket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'socket_plate')
        self.socket_top_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'socket_top_center')
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        self.moving_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'moving_jaw_so101_v1')

        if (not task.bolted) or self.ood_task.spring_mount:
            cur_mass = float(self.model.body_mass[self.socket_body_id])
            scale = float(task.socket_mass_kg) / max(cur_mass, 1e-9)
            self.model.body_mass[self.socket_body_id] *= scale
            self.model.body_inertia[self.socket_body_id] *= scale
            mujoco.mj_setConst(self.model, self.data)

        self._peg_geoms = frozenset(
            i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) == self.peg_body_id
        )
        self._socket_geoms = frozenset(
            i for i in range(self.model.ngeom) if int(self.model.geom_bodyid[i]) == self.socket_body_id
        )
        self._arm_root_id = int(self.model.body_rootid[self.gripper_body_id])
        self._hole_outer_r = _hole_outer_radius_ood(task.shape)
        self._geom_spec = AssemblyGeometrySpec(shape='round', clearance_m=task.clearance_m)
        self._cforce_buf = np.zeros(6, dtype=float)
        self._socket_init_pos = np.zeros(3, dtype=float)

    def _compute_privileged_info(self) -> dict:
        peg_tip = self.data.site_xpos[self.peg_tip_site_id].copy()
        sock_top = self.data.site_xpos[self.socket_top_site_id].copy()
        R_sock = self.data.xmat[self.socket_body_id].reshape(3, 3)

        # Compute depth along socket's actual local normal & lateral error in socket frame (exact for tilted sockets)
        rel_world = peg_tip - sock_top
        rel_local = R_sock.T @ rel_world
        xy_err = rel_local[:2].copy()
        radial_err = float(np.linalg.norm(xy_err))
        peg_depth = float(-rel_local[2])

        if self.ood_task.spring_mount:
            sock_disp_mm = float(np.linalg.norm(self.data.xpos[self.socket_body_id] - self._socket_init_pos) * 1000.0)
            sock_tilt_deg = 0.0
        elif not self.task.bolted:
            sock_xy_now = self.data.xpos[self.socket_body_id, :2]
            sock_disp_mm = float(np.linalg.norm(sock_xy_now - self._socket_init_xy) * 1000.0)
            r_zz = float(R_sock[2, 2])
            sock_tilt_deg = float(np.degrees(np.arccos(np.clip(r_zz, -1.0, 1.0))))
        else:
            sock_disp_mm = 0.0
            sock_tilt_deg = 0.0

        if sock_disp_mm > self.episode_max_socket_disp_mm:
            self.episode_max_socket_disp_mm = sock_disp_mm
        if sock_tilt_deg > self.episode_max_socket_tilt_deg:
            self.episode_max_socket_tilt_deg = sock_tilt_deg
        if self._step_peak_contact_N > self.episode_peak_contact_force_N:
            self.episode_peak_contact_force_N = self._step_peak_contact_N

        if peg_depth >= TARGET_SEAT_DEPTH_M and radial_err < self._hole_outer_r:
            c_class = 'seated'
        elif peg_depth >= 0.002 and radial_err < self._hole_outer_r:
            c_class = 'entering'
        elif self._step_peg_socket_contact:
            c_class = 'rim_contact' if radial_err <= self._hole_outer_r + 0.003 else 'plate_contact'
        else:
            c_class = 'free'

        return {
            'socket_xy_error': (peg_tip[:2] - sock_top[:2]).copy(),
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

    def reset(self, task=None, servo_cfg=None) -> tuple[dict, dict]:
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

        # Place peg tip above actual shifted/tilted plate top (with extra margin for tilted plate/peg)
        tilt_extra_z = 0.0025 if (abs(self.ood_task.socket_tilt_deg) > 1e-6 or abs(self.ood_task.grasp_tilt_deg) > 1e-6) else 0.0
        start_z = self._geom_spec.plate_top_z_m + self.ood_task.socket_dz_m + self.task.start_clearance_z_m + tilt_extra_z
        start_xyz = np.array([
            self.task.socket_xy[0] + self.task.misalign_xy[0],
            self.task.socket_xy[1] + self.task.misalign_xy[1],
            start_z,
        ], dtype=float)
        q0 = solve_vertical_ik(self.nominal_model, self.nominal_data, self.nominal_tip_site, start_xyz)
        self.data.qpos[:6] = q0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:6] = q0
        if (not self.task.bolted) and (not self.ood_task.spring_mount):
            z_plate = self._geom_spec.bottom_thickness_m + self._geom_spec.plate_thickness_m / 2.0 + self.ood_task.socket_dz_m
            self.data.qpos[6:9] = [self.task.socket_xy[0], self.task.socket_xy[1], z_plate]
            self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

        for _ in range(60):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._socket_init_xy = self.data.xpos[self.socket_body_id, :2].copy()
        self._socket_init_pos = self.data.xpos[self.socket_body_id].copy()
        self._q_cmd = q0.copy()
        self._q_backlash = q0.copy()
        q_quant = self._quantize_q(self.data.qpos[:6])
        self._q_quant_prev = q_quant.copy()
        self._prev_action = np.zeros(8, dtype=float)
        self._force_lag_buf.clear()
        raw_f = self.data.actuator_force[:6].copy()
        for _ in range(max(self.servo_cfg.force_lag_steps + 1, 1)):
            f_noisy = self._read_noisy_lagged_force(raw_f)
        obs = self._build_obs(q_quant, np.zeros(6, dtype=float), f_noisy)
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
            'episode_max_socket_disp_mm': 0.0,
            'episode_max_socket_tilt_deg': 0.0,
            'privileged': priv,
        }
        return obs, info


def build_phase6_ood_suites() -> dict[str, list[OODTaskSpec]]:
    """Build the 5 deterministic Phase 6 OOD Stress Suites (20 trials each, disjoint seeds 5100..5519)."""
    suites: dict[str, list[OODTaskSpec]] = {}

    def _base_xy(rng: np.random.Generator) -> tuple[float, float]:
        return (float(0.220 + rng.uniform(-0.010, 0.010)), float(rng.uniform(-0.012, 0.012)))

    # 1. OOD_Height_Tilt: +-12 mm height shift AND 3-5 deg tilt (bolted)
    rng = np.random.default_rng(5100)
    s1: list[OODTaskSpec] = []
    for i in range(20):
        sh = ALL_SHAPES[i % len(ALL_SHAPES)]
        r = float(rng.uniform(0.0060, 0.0145))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        dz = 0.012 if (i % 2 == 0) else -0.012
        t_deg = float(rng.uniform(3.0, 5.0))
        t_phi = float(rng.uniform(0.0, 2.0 * math.pi))
        s1.append(OODTaskSpec(
            base=InsertionTaskSpec(
                trial_id=f'p6-ht-{i:02d}', seed=5100 + i, shape=sh,
                clearance_m=float(rng.uniform(0.0010, 0.0015)), bolted=True,
                socket_xy=_base_xy(rng), misalign_xy=(r * math.cos(th), r * math.sin(th)),
                start_clearance_z_m=float(rng.uniform(0.0015, 0.0025)),
                grasp_offset_xy=(float(rng.uniform(-0.0010, 0.0010)), float(rng.uniform(-0.0010, 0.0010))),
                mu_peg_hole=float(rng.uniform(0.20, 0.38)), mu_table=0.45,
                socket_mass_kg=0.250, backlash_rad=float(rng.uniform(0.003, 0.006)), max_steps=250,
            ),
            suite='OOD_Height_Tilt', socket_dz_m=dz, socket_tilt_deg=t_deg, socket_tilt_phi=t_phi,
        ))
    suites['OOD_Height_Tilt'] = s1

    # 2. OOD_Spring_Mount: compliant 3-axis spring-damper (k_z=800 N/m, k_xy=600 N/m)
    rng = np.random.default_rng(5200)
    s2: list[OODTaskSpec] = []
    for i in range(20):
        sh = ALL_SHAPES[i % len(ALL_SHAPES)]
        r = float(rng.uniform(0.0060, 0.0145))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        s2.append(OODTaskSpec(
            base=InsertionTaskSpec(
                trial_id=f'p6-spr-{i:02d}', seed=5200 + i, shape=sh,
                clearance_m=float(rng.uniform(0.0010, 0.0015)), bolted=True,
                socket_xy=_base_xy(rng), misalign_xy=(r * math.cos(th), r * math.sin(th)),
                start_clearance_z_m=float(rng.uniform(0.0015, 0.0025)),
                grasp_offset_xy=(float(rng.uniform(-0.0010, 0.0010)), float(rng.uniform(-0.0010, 0.0010))),
                mu_peg_hole=float(rng.uniform(0.20, 0.38)), mu_table=0.45,
                socket_mass_kg=0.250, backlash_rad=float(rng.uniform(0.003, 0.006)), max_steps=250,
            ),
            suite='OOD_Spring_Mount', spring_mount=True, spring_k_xy=600.0, spring_k_z=800.0, spring_damping=16.0,
        ))
    suites['OOD_Spring_Mount'] = s2

    # 3. OOD_NonConvex_Asym: L-shape (10) + keyed_notch (10) at 1.0 mm clearance (50% bolted / 50% unbolted)
    rng = np.random.default_rng(5300)
    nc_shapes = ('l_shape', 'keyed_notch')
    s3: list[OODTaskSpec] = []
    for i in range(20):
        sh = nc_shapes[i % 2]
        r = float(rng.uniform(0.0060, 0.0145))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        s3.append(OODTaskSpec(
            base=InsertionTaskSpec(
                trial_id=f'p6-nc-{i:02d}', seed=5300 + i, shape=sh,
                clearance_m=0.0010, bolted=(i % 4 < 2),
                socket_xy=_base_xy(rng), misalign_xy=(r * math.cos(th), r * math.sin(th)),
                start_clearance_z_m=float(rng.uniform(0.0015, 0.0025)),
                grasp_offset_xy=(float(rng.uniform(-0.0010, 0.0010)), float(rng.uniform(-0.0010, 0.0010))),
                mu_peg_hole=float(rng.uniform(0.20, 0.38)), mu_table=float(rng.uniform(0.35, 0.55)),
                socket_mass_kg=float(rng.uniform(0.180, 0.320)), backlash_rad=float(rng.uniform(0.003, 0.006)), max_steps=250,
            ),
            suite='OOD_NonConvex_Asym',
        ))
    suites['OOD_NonConvex_Asym'] = s3

    # 4. OOD_Grasp_Error: +-3.5 mm grasp offset AND 4-6 deg in-hand peg tilt (unobserved)
    rng = np.random.default_rng(5400)
    s4: list[OODTaskSpec] = []
    for i in range(20):
        sh = ALL_SHAPES[i % len(ALL_SHAPES)]
        r = float(rng.uniform(0.0060, 0.0140))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        g_th = float(rng.uniform(0.0, 2.0 * math.pi))
        g_r = 0.0035
        g_tilt = float(rng.uniform(4.0, 6.0))
        g_phi = float(rng.uniform(0.0, 2.0 * math.pi))
        s4.append(OODTaskSpec(
            base=InsertionTaskSpec(
                trial_id=f'p6-gr-{i:02d}', seed=5400 + i, shape=sh,
                clearance_m=float(rng.uniform(0.0012, 0.0015)), bolted=(i % 2 == 0),
                socket_xy=_base_xy(rng), misalign_xy=(r * math.cos(th), r * math.sin(th)),
                start_clearance_z_m=float(rng.uniform(0.0015, 0.0025)),
                grasp_offset_xy=(g_r * math.cos(g_th), g_r * math.sin(g_th)),
                mu_peg_hole=float(rng.uniform(0.18, 0.35)), mu_table=float(rng.uniform(0.35, 0.55)),
                socket_mass_kg=float(rng.uniform(0.180, 0.320)), backlash_rad=float(rng.uniform(0.003, 0.006)), max_steps=250,
            ),
            suite='OOD_Grasp_Error', grasp_tilt_deg=g_tilt, grasp_tilt_phi=g_phi,
        ))
    suites['OOD_Grasp_Error'] = s4

    # 5. OOD_UltraTight_0.30mm: sub-encoder-count 0.30 mm clearance across seen + unseen shapes
    rng = np.random.default_rng(5500)
    s5: list[OODTaskSpec] = []
    for i in range(20):
        sh = ALL_SHAPES[i % len(ALL_SHAPES)]
        r = float(rng.uniform(0.0040, 0.0135))
        th = float(rng.uniform(0.0, 2.0 * math.pi))
        s5.append(OODTaskSpec(
            base=InsertionTaskSpec(
                trial_id=f'p6-ut-{i:02d}', seed=5500 + i, shape=sh,
                clearance_m=0.00030, bolted=(i % 2 == 0),
                socket_xy=_base_xy(rng), misalign_xy=(r * math.cos(th), r * math.sin(th)),
                start_clearance_z_m=float(rng.uniform(0.0015, 0.0025)),
                grasp_offset_xy=(float(rng.uniform(-0.0010, 0.0010)), float(rng.uniform(-0.0010, 0.0010))),
                mu_peg_hole=float(rng.uniform(0.18, 0.35)), mu_table=float(rng.uniform(0.35, 0.55)),
                socket_mass_kg=float(rng.uniform(0.180, 0.320)), backlash_rad=float(rng.uniform(0.003, 0.006)), max_steps=250,
            ),
            suite='OOD_UltraTight_0.30mm',
        ))
    suites['OOD_UltraTight_0.30mm'] = s5

    return suites


def verify_phase6_disjointness(suites: dict[str, list[OODTaskSpec]]) -> None:
    existing_seeds = {
        t.seed
        for fn in (build_regime_a_bolted, build_regime_b_unbolted, build_split_c_tight_0_5mm, build_slippery_unbolted_ood_suite)
        for t in fn()
    }
    seen: set[int] = set()
    for sk, trials in suites.items():
        assert len(trials) == 20, f'{sk}: expected 20 trials, got {len(trials)}'
        for ood in trials:
            s = ood.base.seed
            assert s < TRAIN_SEED_OFFSET, f'{sk}: seed {s} overlaps train offset'
            assert s not in existing_seeds, f'{sk}: seed {s} overlaps existing eval suites'
            assert s not in seen, f'{sk}: duplicate seed {s}'
            seen.add(s)


def _eval_ood_episode_worker(job: dict) -> dict:
    """Evaluate one deterministic OOD trial for either a baseline controller or a frozen Phase 5 policy."""
    ood: OODTaskSpec = job['ood']
    task = ood.base
    kind: str = job['kind']  # 'baseline' | 'rl'

    env = Phase6OODEnv(ood_task=ood, render=False)
    try:
        obs, info = env.reset()
        terminated = False
        truncated = False
        max_depth_mm = float(info['privileged']['peg_depth'] * 1000.0)
        hole_r = _hole_outer_radius_ood(task.shape)
        went_off_plate = False

        if kind == 'baseline':
            cname = job['controller']
            if cname == 'BlindSpiral_Stiff_1.0':
                ctrl = BlindSpiralController(torque_scale=1.0)
                use_gc = False
            elif cname == 'BlindSpiral_GravComp_0.35':
                ctrl = BlindSpiralController(torque_scale=1.0)
                use_gc = True
            elif cname == 'Privileged_Oracle':
                ctrl = PrivilegedOracleController()
                use_gc = False
            else:
                raise ValueError(f'Unknown baseline {cname}')
            ctrl.reset(obs)
            while not (terminated or truncated):
                act = ctrl.act(obs, info)
                if use_gc:
                    env_act = {'delta_xyz': act[:3], 'gravcomp_alpha': BASE_SPIRAL_ALPHA}
                else:
                    env_act = act
                obs, _, terminated, truncated, info = env.step(env_act)
                cur_p = info['privileged']
                max_depth_mm = max(max_depth_mm, float(cur_p['peg_depth'] * 1000.0))
                cur_xy_v = np.asarray(cur_p['socket_xy_error'], dtype=float)
                if float(np.max(np.abs(cur_xy_v))) > P2_PLATE_HARD_HALF_M or (
                    float(cur_p['peg_depth']) > 0.0020 and float(np.linalg.norm(cur_xy_v)) > hole_r + 0.0030
                ):
                    went_off_plate = True
            aux_acc = 0.0
            dir_cos = 0.0
            g_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
            a_lat_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
            a_arm_by_phase: dict[str, list[float]] = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
        else:
            mode: str = job['mode']
            base_ctrl = PausableBaseSpiralController(torque_scale=1.0)
            base_ctrl.reset(obs)
            builder = Phase4ResidualBuilder(mode=mode)
            pol = Phase5ResidualPPO(
                obs_dim=builder.obs_dim,
                mode=mode,
                base_alpha=BASE_SPIRAL_ALPHA,
                critic_dim=31,
                hidden_dim=64,
            )
            pol.set_weights(job['actor_weights'])
            h = pol.init_hidden()
            aux_correct = 0
            aux_total = 0
            dir_cos_list: list[float] = []
            g_by_phase = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
            a_lat_by_phase = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}
            a_arm_by_phase = {'free': [], 'plate': [], 'rim': [], 'insert_seat': []}

            while not (terminated or truncated):
                base_dxyz = base_ctrl.act_base(obs)
                base_r_m = min((base_ctrl.pitch_m / (2.0 * math.pi)) * base_ctrl.theta, base_ctrl.max_radius_m)
                x_vec = builder.step(obs, base_dxyz, base_ctrl.phase, base_r_m, base_ctrl.insert_stall_steps)
                true_cls_name = info['privileged']['contact_class']
                true_cls = CONTACT_CLASS_TO_IDX[true_cls_name]
                env_act, _, g_spiral, alpha_pair, _, h, pred_cls, pred_dir = pol.act_step_p4(
                    x_vec, h, base_dxyz, deterministic=True
                )
                base_ctrl.set_spiral_gate(g_spiral)
                builder.set_prev_controls(g_spiral, alpha_pair)
                aux_correct += int(pred_cls == true_cls)
                aux_total += 1
                xy_e = np.asarray(info['privileged']['socket_xy_error'], dtype=float)
                xy_n = float(np.linalg.norm(xy_e))
                pn = float(np.linalg.norm(pred_dir))
                if xy_n > 5e-4 and pn > 1e-6:
                    dir_cos_list.append(float(np.dot(pred_dir / pn, -xy_e / xy_n)))

                obs, _, terminated, truncated, info = env.step(env_act)
                p_key = (
                    'plate' if true_cls_name == 'plate_contact'
                    else 'rim' if true_cls_name == 'rim_contact'
                    else 'insert_seat' if true_cls_name in ('entering', 'seated')
                    else 'free'
                )
                g_by_phase[p_key].append(g_spiral)
                if alpha_pair is not None:
                    a_lat_by_phase[p_key].append(alpha_pair[0])
                    a_arm_by_phase[p_key].append(alpha_pair[1])

                cur_p = info['privileged']
                max_depth_mm = max(max_depth_mm, float(cur_p['peg_depth'] * 1000.0))
                cur_xy_v = np.asarray(cur_p['socket_xy_error'], dtype=float)
                if float(np.max(np.abs(cur_xy_v))) > P2_PLATE_HARD_HALF_M or (
                    float(cur_p['peg_depth']) > 0.0020 and float(np.linalg.norm(cur_xy_v)) > hole_r + 0.0030
                ):
                    went_off_plate = True
            aux_acc = aux_correct / max(aux_total, 1)
            dir_cos = float(np.mean(dir_cos_list)) if dir_cos_list else 0.0

        priv = info['privileged']
        return {
            'trial_id': task.trial_id,
            'seed': task.seed,
            'shape': task.shape,
            'split': 'seen' if task.shape in SEEN_SHAPES else 'unseen',
            'bolted': task.bolted,
            'clearance_mm': round(task.clearance_m * 1000.0, 3),
            'misalign_mm': round(float(np.linalg.norm(task.misalign_xy) * 1000.0), 2),
            'success': bool(info['success']),
            'failure_mode': info['failure_mode'],
            'went_off_plate': bool(went_off_plate),
            'steps': int(env.step_count),
            'time_s': round(float(info['time_s']), 3),
            'final_depth_mm': round(float(priv['peg_depth'] * 1000.0), 2),
            'max_depth_mm': round(max_depth_mm, 2),
            'peak_contact_force_N': round(float(info['episode_peak_contact_force_N']), 2),
            'max_socket_disp_mm': round(float(info['episode_max_socket_disp_mm']), 2),
            'max_socket_tilt_deg': round(float(info['episode_max_socket_tilt_deg']), 2),
            'self_collision_steps': int(info['self_collision_steps']),
            'joint_limit_steps': int(info['joint_limit_steps']),
            'aux_contact_acc': aux_acc,
            'aux_dir_cosine': dir_cos,
            'ts_free': None, 'ts_search_rim': None, 'ts_insert_seat': None,
            'g_spiral_free': float(np.mean(g_by_phase['free'])) if g_by_phase['free'] else None,
            'g_spiral_plate': float(np.mean(g_by_phase['plate'])) if g_by_phase['plate'] else None,
            'g_spiral_rim': float(np.mean(g_by_phase['rim'])) if g_by_phase['rim'] else None,
            'g_spiral_insert_seat': float(np.mean(g_by_phase['insert_seat'])) if g_by_phase['insert_seat'] else None,
            'alpha_lat_free': float(np.mean(a_lat_by_phase['free'])) if a_lat_by_phase['free'] else None,
            'alpha_arm_free': float(np.mean(a_arm_by_phase['free'])) if a_arm_by_phase['free'] else None,
            'alpha_lat_plate': float(np.mean(a_lat_by_phase['plate'])) if a_lat_by_phase['plate'] else None,
            'alpha_arm_plate': float(np.mean(a_arm_by_phase['plate'])) if a_arm_by_phase['plate'] else None,
            'alpha_lat_rim': float(np.mean(a_lat_by_phase['rim'])) if a_lat_by_phase['rim'] else None,
            'alpha_arm_rim': float(np.mean(a_arm_by_phase['rim'])) if a_arm_by_phase['rim'] else None,
            'alpha_lat_insert_seat': float(np.mean(a_lat_by_phase['insert_seat'])) if a_lat_by_phase['insert_seat'] else None,
            'alpha_arm_insert_seat': float(np.mean(a_arm_by_phase['insert_seat'])) if a_arm_by_phase['insert_seat'] else None,
        }
    finally:
        env.close()
        del env
        gc.collect()
        _malloc_trim()


def main() -> None:
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--ckpt-dir', type=str, default=str(THIS_DIR / 'phase5_checkpoints'))
    parser.add_argument('--out', type=str, default=str(THIS_DIR / 'phase6_ood_results.json'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--oracle-only', action='store_true')
    args = parser.parse_args()

    t0 = time.perf_counter()
    ckpt_dir = Path(args.ckpt_dir)
    suites = build_phase6_ood_suites()
    verify_phase6_disjointness(suites)

    baseline_names = ('Privileged_Oracle',) if args.oracle_only else (
        'Privileged_Oracle',
        'BlindSpiral_Stiff_1.0',
        'BlindSpiral_GravComp_0.35',
    )
    rl_specs = () if args.oracle_only else (
        ('R1_v2_Res_PosOnly', 'res_pos_only'),
        ('R2_v2_Res_GravComp_TL', 'res_gravcomp_tl'),
    )
    for label, _ in rl_specs:
        for s in args.seeds:
            p = ckpt_dir / f'{label}_seed{s}.npz'
            if not p.exists():
                raise FileNotFoundError(f'Missing frozen Phase 5 checkpoint: {p}')

    print(
        f'Phase 6 Zero-Shot OOD Stress Evaluation ({len(suites)} suites x 20 trials = 100 OOD trials/model, '
        f'{args.workers} workers, ZERO retraining, ZERO policy edits)...',
        flush=True,
    )

    baselines: dict[str, dict] = {}
    rl_runs: dict[str, list[dict]] = {lbl: [] for lbl, _ in rl_specs}
    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=args.workers, initializer=_init_worker_single_thread) as pool:
        print('\n=== 1. Reference Baselines & Oracle ===', flush=True)
        for cname in baseline_names:
            baselines[cname] = {}
            parts = []
            for sk in OOD_SUITE_ORDER:
                jobs = [{'kind': 'baseline', 'controller': cname, 'ood': o} for o in suites[sk]]
                rows = pool.map(_eval_ood_episode_worker, jobs)
                summ = summarize_suite_p4(rows)
                baselines[cname][sk] = {'summary': summ, 'episodes': rows}
                parts.append(
                    f'{sk[4:10]}={summ["successes"]:2d}/20({100*summ["success_rate"]:5.1f}%,'
                    f'F={summ["median_peak_contact_force_N"]:4.1f}N,col={summ["self_collision_steps"]},jl={summ["joint_limit_steps"]})'
                )
            print(f'  {cname:26s} | ' + ' | '.join(parts), flush=True)

        if rl_specs:
            print('\n=== 2. Frozen Phase 5 Checkpoints (3 Seeds Each, Zero Retraining) ===', flush=True)
            for label, mode in rl_specs:
                for s in args.seeds:
                    ckpt_path = ckpt_dir / f'{label}_seed{s}.npz'
                    pol = Phase5ResidualPPO.load(ckpt_path)
                    aw = pol.get_actor_weights()
                    eval_out = {}
                    parts = []
                    for sk in OOD_SUITE_ORDER:
                        jobs = [{'kind': 'rl', 'mode': mode, 'actor_weights': aw, 'ood': o} for o in suites[sk]]
                        rows = pool.map(_eval_ood_episode_worker, jobs)
                        summ = summarize_suite_p4(rows)
                        eval_out[sk] = {'summary': summ, 'episodes': rows}
                        parts.append(
                            f'{sk[4:10]}={summ["successes"]:2d}/20({100*summ["success_rate"]:5.1f}%,'
                            f'F={summ["median_peak_contact_force_N"]:4.1f}N,rim={summ["jammed_on_rim_count"]:2d})'
                        )
                    print(f'  {label:22s} s={s} | ' + ' | '.join(parts), flush=True)
                    rl_runs[label].append({
                        'meta': {'label': label, 'mode': mode, 'seed': s, 'checkpoint': str(ckpt_path.relative_to(THIS_DIR))},
                        'eval': eval_out,
                    })

    if args.oracle_only:
        print(f'\nOracle-only solvability gate finished in {time.perf_counter() - t0:.1f}s.', flush=True)
        return

    rl_aggregate = {}
    for label, mode in rl_specs:
        runs = rl_runs[label]
        rl_aggregate[label] = {
            'mode': mode,
            'seeds': list(args.seeds),
            'eval': {sk: aggregate_seeds_p4(runs, sk) for sk in OOD_SUITE_ORDER},
        }

    total_col = sum(
        v['eval'][sk]['total_self_collision_steps'] for v in rl_aggregate.values() for sk in OOD_SUITE_ORDER
    ) + sum(
        baselines[b][sk]['summary']['self_collision_steps'] for b in baselines for sk in OOD_SUITE_ORDER
    )
    total_jlim = sum(
        v['eval'][sk]['total_joint_limit_steps'] for v in rl_aggregate.values() for sk in OOD_SUITE_ORDER
    ) + sum(
        baselines[b][sk]['summary']['joint_limit_steps'] for b in baselines for sk in OOD_SUITE_ORDER
    )
    total_rollouts = (len(baseline_names) + len(rl_specs) * len(args.seeds)) * len(suites) * 20

    report = {
        'total_wall_s': round(time.perf_counter() - t0, 2),
        'parent_rss_mb': round(get_rss_mb(None), 1),
        'total_eval_rollouts': total_rollouts,
        'total_eval_self_collision_steps': total_col,
        'total_eval_joint_limit_steps': total_jlim,
        'suite_order': list(OOD_SUITE_ORDER),
        'baselines': baselines,
        'rl_aggregate': rl_aggregate,
        'rl_runs': rl_runs,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2) + '\n')
    print(
        f'\n=== PHASE 6 OOD STRESS AUDIT FINISHED ({report["total_wall_s"]:.1f}s, {total_rollouts} rollouts) '
        f'| self_col={total_col}, jlim={total_jlim} ===',
        flush=True,
    )
    print(f'Saved results to {out_path}', flush=True)


if __name__ == '__main__':
    main()
