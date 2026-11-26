"""Exact convex peg-in-hole assembly geometry via mjfit + mandatory gravity drop-test gate.

Uses `mjfit` (exact convex prism decomposition of planar outlines, zero CoACD/V-HACD,
zero mesh SDF) to guarantee exact per-side clearance with zero wall intrusion into the hole.

Disjoint shape splits:
- Seen / Train shapes:  round, square, rectangle
- Held-Out Unseen:      hex, triangle, d_shape
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET
import mjfit
import mujoco
import numpy as np

SEEN_SHAPES: tuple[str, ...] = ('round', 'square', 'rectangle')
UNSEEN_SHAPES: tuple[str, ...] = ('hex', 'triangle', 'd_shape')
ALL_SHAPES: tuple[str, ...] = SEEN_SHAPES + UNSEEN_SHAPES

DEFAULT_PLATE_SIZE_M = 0.084       # 84 mm x 84 mm tabletop socket footprint
DEFAULT_PLATE_THICKNESS_M = 0.020  # 20 mm hole depth (target seated depth >= 12 mm)
DEFAULT_BOTTOM_THICKNESS_M = 0.004 # 4 mm solid base (total socket height = 24 mm)
DEFAULT_PEG_LENGTH_M = 0.036       # 36 mm peg (bottom 20 mm inserts, top 16 mm in jaws)
DEFAULT_PEG_MASS_KG = 0.025


@dataclass(frozen=True)
class AssemblyGeometrySpec:
    """Exact convex assembly specification for a peg+socket pair."""
    shape: str = 'round'
    clearance_m: float = 0.0012
    peg_length_m: float = DEFAULT_PEG_LENGTH_M
    plate_size_m: float = DEFAULT_PLATE_SIZE_M
    plate_thickness_m: float = DEFAULT_PLATE_THICKNESS_M
    bottom_thickness_m: float = DEFAULT_BOTTOM_THICKNESS_M
    peg_mass_kg: float = DEFAULT_PEG_MASS_KG

    @property
    def plate_top_z_m(self) -> float:
        return self.bottom_thickness_m + self.plate_thickness_m

    @property
    def hole_floor_z_m(self) -> float:
        return self.bottom_thickness_m

    @property
    def is_seen_shape(self) -> bool:
        return self.shape in SEEN_SHAPES


def _d_shape_polygon(radius: float = 0.0095, flat_x: float = 0.0055, n: int = 20) -> np.ndarray:
    """Convex D-shaped cross-section: circle of radius clipped by vertical chord x <= flat_x."""
    if not (0.0 < flat_x < radius):
        raise ValueError(f'flat_x ({flat_x}) must lie strictly inside (0, radius={radius})')
    a_max = math.acos(flat_x / radius)
    angles = np.linspace(a_max, 2.0 * math.pi - a_max, n)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return pts


def build_shape_polygon(shape: str) -> np.ndarray:
    """Return counterclockwise (N, 2) hole outline in meters for one of the 6 benchmark shapes."""
    if shape == 'round':
        # 18 mm diameter circular hole; max_error = 0.05 mm << 0.5 mm tightest clearance.
        return mjfit.shapes.circle(radius=0.0090, max_error=5e-5)
    if shape == 'square':
        return mjfit.shapes.rect(width=0.0170, height=0.0170)
    if shape == 'rectangle':
        return mjfit.shapes.rect(width=0.0200, height=0.0140)
    if shape == 'hex':
        return mjfit.shapes.ngon(6, radius=0.0100)
    if shape == 'triangle':
        return mjfit.shapes.ngon(3, radius=0.0125)
    if shape == 'd_shape':
        return _d_shape_polygon(radius=0.0095, flat_x=0.0055, n=20)
    raise ValueError(f'Unknown shape {shape!r}; expected one of {ALL_SHAPES}')


def hole_outer_radius_m(shape: str) -> float:
    """Maximum radial extent of the centered hole polygon in meters."""
    poly = build_shape_polygon(shape)
    poly = poly - poly.mean(axis=0)
    return float(np.max(np.linalg.norm(poly, axis=1)))


def generate_mjfit_parts_xml(
    spec: AssemblyGeometrySpec,
    friction: float = 0.30,
    free_plate: bool = False,
    standalone: bool = False,
    spawn_height_m: float = 0.001,
    timestep: float = 0.001,
) -> str:
    """Generate exact convex MJCF for peg + socket via mjfit.peg."""
    hole = build_shape_polygon(spec.shape)
    return mjfit.peg(
        hole=hole,
        clearance=spec.clearance_m,
        peg_length=spec.peg_length_m,
        plate_size=spec.plate_size_m,
        plate_thickness=spec.plate_thickness_m,
        bottom_thickness=spec.bottom_thickness_m,
        peg_mass=spec.peg_mass_kg,
        plate_density=1500.0,
        spawn_height=spawn_height_m,
        center=True,
        include_peg=True,
        free_plate=free_plate,
        friction=(float(friction), 0.005, 0.0005),
        condim=4,
        timestep=timestep,
        standalone=standalone,
        floor=standalone,
    )


def extract_mjfit_elements(
    spec: AssemblyGeometrySpec,
    friction: float = 0.30,
) -> tuple[list[ET.Element], list[ET.Element], list[ET.Element]]:
    """Return (mesh_elements, plate_geom_elements, peg_geom_elements) from mjfit."""
    xml = generate_mjfit_parts_xml(spec, friction=friction, free_plate=False, standalone=False)
    root = ET.fromstring(xml)
    asset = root.find('asset')
    wb = root.find('worldbody')
    assert asset is not None and wb is not None

    meshes = list(asset.findall('mesh'))
    plate_body = wb.find("./body[@name='plate']")
    peg_body = wb.find("./body[@name='peg']")
    assert plate_body is not None and peg_body is not None

    plate_geoms = list(plate_body.findall('geom'))
    peg_geoms = list(peg_body.findall('geom'))
    return meshes, plate_geoms, peg_geoms


def run_drop_test(
    shape: str,
    clearance_m: float,
    sim_duration_s: float = 0.6,
    timestep: float = 0.001,
) -> dict:
    """Mandatory drop-test gate: peg centered 1 mm above hole falls cleanly to full seated depth."""
    spec = AssemblyGeometrySpec(shape=shape, clearance_m=clearance_m)
    xml = generate_mjfit_parts_xml(
        spec,
        friction=0.30,
        free_plate=False,
        standalone=True,
        spawn_height_m=0.001,
        timestep=timestep,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    peg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'peg')
    n_steps = int(round(sim_duration_s / timestep))

    max_xy_drift_m = 0.0
    max_tilt_deg = 0.0
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        xy_drift = float(np.linalg.norm(data.xpos[peg_id, :2]))
        r_zz = float(data.xmat[peg_id].reshape(3, 3)[2, 2])
        tilt_deg = float(np.degrees(np.arccos(np.clip(r_zz, -1.0, 1.0))))
        if xy_drift > max_xy_drift_m:
            max_xy_drift_m = xy_drift
        if tilt_deg > max_tilt_deg:
            max_tilt_deg = tilt_deg

    z_bottom = float(data.xpos[peg_id, 2]) - spec.peg_length_m / 2.0
    depth_m = spec.plate_top_z_m - z_bottom
    target_full_depth_m = spec.plate_thickness_m - 0.0005  # >= 19.5 mm out of 20.0 mm
    passed = bool(depth_m >= target_full_depth_m and max_tilt_deg < 3.0)
    return {
        'shape': shape,
        'split': 'seen' if shape in SEEN_SHAPES else 'unseen',
        'clearance_mm': round(clearance_m * 1000.0, 3),
        'depth_mm': round(depth_m * 1000.0, 3),
        'target_depth_mm': round(target_full_depth_m * 1000.0, 3),
        'max_xy_drift_mm': round(max_xy_drift_m * 1000.0, 3),
        'max_tilt_deg': round(max_tilt_deg, 3),
        'passed': passed,
    }


def run_all_drop_tests(clearances_m: tuple[float, ...] = (0.0010, 0.0005)) -> list[dict]:
    """Run the drop-test gate across all 6 shapes at every requested clearance."""
    results = []
    for c in clearances_m:
        for shape in ALL_SHAPES:
            results.append(run_drop_test(shape, c))
    return results
