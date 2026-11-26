# SPDX-License-Identifier: MIT
"""3-minute static gravity-torque calibration for a physical SO-101 arm (with `--dry-run` mode).

Sweeps safe static poses spanning the SO-101 workspace, reads STS3215 `Present_Load (0x3C)`
(or simulated static torques in `--dry-run`), fits per-joint linear scale & offset against
`GravCompTL.compute_gravity_torque(q)` via least squares, and saves `so101_gravcomp_calib.json`.

Usage:
    python -m gravcomp_tl.calibrate_so101 --dry-run --out so101_gravcomp_calib.json
    python -m gravcomp_tl.calibrate_so101 --port /dev/ttyACM0 --out so101_gravcomp_calib.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from .core import JOINT_NAMES, GravCompTL, rad_to_sts3215_ticks, sts3215_ticks_to_rad
from .sts3215_driver import DryRunSTS3215Bus, FeetechSCServoBus

# 12 safe static calibration poses (rad) inside SO-101 joint limits, exciting gravity on lift/elbow/wrist
CALIBRATION_POSES_RAD: tuple[tuple[float, ...], ...] = (
    (0.00, -0.45,  0.20,  0.80,  0.00),
    (0.00, -0.20,  0.40,  1.10,  0.00),
    (0.00,  0.10,  0.65,  1.25,  0.00),
    (0.00,  0.35,  0.25,  0.90,  0.00),
    (0.30, -0.35,  0.50,  1.05,  0.25),
    (-0.30, -0.35, 0.50,  1.05, -0.25),
    (0.20,  0.00,  0.85,  0.60,  0.00),
    (-0.20, 0.00,  0.85,  0.60,  0.00),
    (0.00, -0.55,  0.75,  0.40,  0.00),
    (0.00,  0.25, -0.10,  1.35,  0.00),
    (0.40, -0.10,  0.30,  1.20,  0.40),
    (-0.40, -0.10, 0.30,  1.20, -0.40),
)


def fit_gravity_calibration(tau_model: np.ndarray, tau_meas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-joint `tau_meas_j ~ scale_j * tau_model_j + offset_j` with safe regularization."""
    n_joints = tau_model.shape[1]
    scales = np.ones(n_joints, dtype=float)
    offsets = np.zeros(n_joints, dtype=float)
    rmses = np.zeros(n_joints, dtype=float)

    for j in range(n_joints):
        x = tau_model[:, j]
        y = tau_meas[:, j]
        if float(np.std(x)) < 1e-3:
            # Joint has near-zero gravity torque (e.g. shoulder_pan vertical axis) -> fit offset only
            scales[j] = 1.0
            offsets[j] = float(np.mean(y - x))
        else:
            A = np.column_stack([x, np.ones_like(x)])
            sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            scales[j] = float(np.clip(sol[0], 0.5, 1.8))
            offsets[j] = float(np.clip(sol[1], -0.35, 0.35))
        pred = scales[j] * x + offsets[j]
        rmses[j] = float(np.sqrt(np.mean((y - pred) ** 2)))
    return scales, offsets, rmses


def run_calibration(
    dry_run: bool = True,
    port: str = '/dev/ttyACM0',
    baudrate: int = 1_000_000,
    settle_s: float = 0.6,
    out_path: str | Path = 'so101_gravcomp_calib.json',
    seed: int = 0,
) -> dict:
    gc = GravCompTL()
    if dry_run:
        bus = DryRunSTS3215Bus(gravity_model=gc)
        rng = np.random.default_rng(seed)
        # Simulate realistic +-8% link-mass mismatch + 0.015 Nm friction bias + load noise in dry-run
        true_scale = np.array([1.0, 1.08, 0.94, 1.05, 1.0], dtype=float)
        true_offset = np.array([0.005, 0.015, -0.010, 0.008, 0.002], dtype=float)
    else:
        bus = FeetechSCServoBus(port=port, baudrate=baudrate)
        rng = None
        true_scale = None
        true_offset = None

    tau_model_rows, tau_meas_rows = [], []
    bus.write_torque_limits_ticks(np.full(5, 650, dtype=np.int32))  # safe 65% torque during static sweep

    try:
        for q_pose in CALIBRATION_POSES_RAD:
            q_arr = np.asarray(q_pose, dtype=float)
            bus.write_goal_positions_ticks(rad_to_sts3215_ticks(q_arr))
            if not dry_run:
                time.sleep(settle_s)
            q_meas = sts3215_ticks_to_rad(bus.read_present_positions_ticks())
            t_mod = gc.compute_gravity_torque(q_meas)
            t_load = bus.read_present_loads_nm()
            if dry_run and rng is not None and true_scale is not None and true_offset is not None:
                t_load = true_scale * t_mod + true_offset + rng.normal(0.0, 0.005, size=5)
            tau_model_rows.append(t_mod)
            tau_meas_rows.append(t_load)
    finally:
        bus.write_torque_limits_ticks(np.zeros(5, dtype=np.int32))
        if hasattr(bus, 'close'):
            bus.close()

    tau_model = np.asarray(tau_model_rows, dtype=float)
    tau_meas = np.asarray(tau_meas_rows, dtype=float)
    scales, offsets, rmses = fit_gravity_calibration(tau_model, tau_meas)

    calib = {
        'dry_run': bool(dry_run),
        'n_poses': len(CALIBRATION_POSES_RAD),
        'joint_names': list(JOINT_NAMES),
        'grav_scale': [round(float(v), 5) for v in scales],
        'grav_offset_nm': [round(float(v), 5) for v in offsets],
        'fit_rmse_nm': [round(float(v), 5) for v in rmses],
        'max_rmse_nm': round(float(np.max(rmses)), 5),
    }
    Path(out_path).write_text(json.dumps(calib, indent=2) + '\n')
    return calib


def main() -> None:
    parser = argparse.ArgumentParser(description='Static SO-101 gravity-torque calibration for GravComp-TL.')
    parser.add_argument('--dry-run', action='store_true', help='Run offline calibration against MuJoCo mock bus.')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0')
    parser.add_argument('--baudrate', type=int, default=1_000_000)
    parser.add_argument('--settle-s', type=float, default=0.6)
    parser.add_argument('--out', type=str, default='so101_gravcomp_calib.json')
    args = parser.parse_args()
    cal = run_calibration(
        dry_run=args.dry_run, port=args.port, baudrate=args.baudrate,
        settle_s=args.settle_s, out_path=args.out,
    )
    print(f"Saved calibration ({cal['n_poses']} poses, max_rmse={cal['max_rmse_nm']:.4f} Nm) -> {args.out}")
    for jn, sc, off, err in zip(cal['joint_names'], cal['grav_scale'], cal['grav_offset_nm'], cal['fit_rmse_nm']):
        print(f'  {jn:14s}: scale={sc:.4f}, offset={off:+.4f} Nm, rmse={err:.4f} Nm')


if __name__ == '__main__':
    main()
