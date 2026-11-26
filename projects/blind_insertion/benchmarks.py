"""Deterministic held-out benchmark suites for Phase 0 & Phase 1 blind insertion.

Two 30-trial deterministic held-out regimes spanning all 6 disjoint shapes
(5 trials/shape = 15 seen + 15 unseen):
- Regime A (`bolted_socket`, seeds 1001..1030): socket welded to table.
- Regime B (`unbolted_socket`, seeds 2001..2030): free socket on table
  (mass in [180, 320] g, table mu in [0.35, 0.55], fails if shoved > 25 mm or tipped > 15 deg).
Plus a dedicated tight-clearance (0.5 mm) verification suite (seeds 3001..3006).
"""
from __future__ import annotations

import math
import numpy as np

from .env import InsertionTaskSpec
from .geometry import ALL_SHAPES, SEEN_SHAPES

REGIME_A_SEED_OFFSET = 1001
REGIME_B_SEED_OFFSET = 2001
TIGHT_SEED_OFFSET = 3001
SPLIT_C_SEED_OFFSET = 3101
TRAIN_SEED_OFFSET = 10000


def _build_30_trials(*, bolted: bool, seed_offset: int, prefix: str) -> list[InsertionTaskSpec]:
    """Generate 30 deterministic trials (5 per shape across all 6 shapes)."""
    rng = np.random.default_rng(seed_offset)
    trials: list[InsertionTaskSpec] = []
    # 5 stratified misalignment radii spanning 6..16 mm (covering the full 6-16 mm range evenly).
    radii_strata_m = [0.0065, 0.0088, 0.0110, 0.0132, 0.0155]
    clearances_strata_m = [0.0010, 0.00112, 0.00125, 0.00138, 0.0015]

    idx = 0
    for s_idx, shape in enumerate(ALL_SHAPES):
        for k in range(5):
            seed = seed_offset + idx
            # Mild jitter within each stratum while staying strictly inside [6, 16] mm.
            r = float(np.clip(radii_strata_m[k] + rng.uniform(-0.0004, 0.0004), 0.006, 0.016))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            misalign = (r * math.cos(theta), r * math.sin(theta))
            clearance = clearances_strata_m[(k + s_idx) % len(clearances_strata_m)]
            socket_xy = (
                float(0.220 + rng.uniform(-0.012, 0.012)),
                float(0.000 + rng.uniform(-0.015, 0.015)),
            )
            grasp_off = (
                float(rng.uniform(-0.0015, 0.0015)),
                float(rng.uniform(-0.0015, 0.0015)),
            )
            mu_ph = float(rng.uniform(0.15, 0.45))
            mu_tab = float(rng.uniform(0.35, 0.55))
            sock_mass = float(rng.uniform(0.180, 0.320))
            backlash = float(rng.uniform(0.003, 0.006))

            split_tag = 'seen' if shape in SEEN_SHAPES else 'unseen'
            trials.append(InsertionTaskSpec(
                trial_id=f'{prefix}-{idx+1:02d}-{shape}-{split_tag}',
                seed=seed,
                shape=shape,
                clearance_m=clearance,
                bolted=bolted,
                socket_xy=socket_xy,
                misalign_xy=misalign,
                start_clearance_z_m=0.002,
                grasp_offset_xy=grasp_off,
                mu_peg_hole=mu_ph,
                mu_table=mu_tab,
                socket_mass_kg=sock_mass,
                backlash_rad=backlash,
                max_steps=250,
            ))
            idx += 1
    return trials


def build_regime_a_bolted() -> list[InsertionTaskSpec]:
    """Regime A: 30 deterministic bolted-socket trials (15 seen + 15 unseen shapes)."""
    return _build_30_trials(bolted=True, seed_offset=REGIME_A_SEED_OFFSET, prefix='regA')


def build_regime_b_unbolted() -> list[InsertionTaskSpec]:
    """Regime B: 30 deterministic unbolted-socket trials (mass 180-320 g, table mu 0.35-0.55)."""
    return _build_30_trials(bolted=False, seed_offset=REGIME_B_SEED_OFFSET, prefix='regB')


def build_tight_clearance_suite() -> list[InsertionTaskSpec]:
    """6-trial tight-clearance (0.5 mm) sanity suite across all 6 shapes."""
    rng = np.random.default_rng(TIGHT_SEED_OFFSET)
    trials: list[InsertionTaskSpec] = []
    for idx, shape in enumerate(ALL_SHAPES):
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        r = 0.010
        trials.append(InsertionTaskSpec(
            trial_id=f'tight-{idx+1:02d}-{shape}',
            seed=TIGHT_SEED_OFFSET + idx,
            shape=shape,
            clearance_m=0.0005,
            bolted=True,
            socket_xy=(0.220, 0.0),
            misalign_xy=(r * math.cos(theta), r * math.sin(theta)),
            start_clearance_z_m=0.002,
            grasp_offset_xy=(0.0, 0.0),
            mu_peg_hole=0.25,
            mu_table=0.45,
            socket_mass_kg=0.250,
            backlash_rad=0.004,
            max_steps=250,
        ))
    return trials


def build_split_c_tight_0_5mm() -> list[InsertionTaskSpec]:
    """Split C (`tight_0.5mm`, n=30): 15 bolted + 15 unbolted at 0.5 mm clearance, 15 seen + 15 unseen."""
    rng = np.random.default_rng(SPLIT_C_SEED_OFFSET)
    radii_strata_m = [0.0065, 0.0088, 0.0110, 0.0132, 0.0155]
    trials: list[InsertionTaskSpec] = []
    idx = 0
    for shape in ALL_SHAPES:
        for k in range(5):
            seed = SPLIT_C_SEED_OFFSET + idx
            # Exact alternating bolted/unbolted -> 15 bolted + 15 unbolted across 30 trials
            # (seen: 8 bolted / 7 unbolted; unseen: 7 bolted / 8 unbolted).
            bolted = (idx % 2 == 0)
            r = float(np.clip(radii_strata_m[k] + rng.uniform(-0.0004, 0.0004), 0.006, 0.016))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            misalign = (r * math.cos(theta), r * math.sin(theta))
            socket_xy = (
                float(0.220 + rng.uniform(-0.012, 0.012)),
                float(0.000 + rng.uniform(-0.015, 0.015)),
            )
            grasp_off = (
                float(rng.uniform(-0.0015, 0.0015)),
                float(rng.uniform(-0.0015, 0.0015)),
            )
            mu_ph = float(rng.uniform(0.15, 0.45))
            mu_tab = float(rng.uniform(0.35, 0.55))
            sock_mass = float(rng.uniform(0.180, 0.320))
            backlash = float(rng.uniform(0.003, 0.006))
            split_tag = 'seen' if shape in SEEN_SHAPES else 'unseen'
            reg_tag = 'B' if bolted else 'U'
            trials.append(InsertionTaskSpec(
                trial_id=f'splitC-{idx+1:02d}-{shape}-{split_tag}-{reg_tag}',
                seed=seed,
                shape=shape,
                clearance_m=0.0005,
                bolted=bolted,
                socket_xy=socket_xy,
                misalign_xy=misalign,
                start_clearance_z_m=0.002,
                grasp_offset_xy=grasp_off,
                mu_peg_hole=mu_ph,
                mu_table=mu_tab,
                socket_mass_kg=sock_mass,
                backlash_rad=backlash,
                max_steps=250,
            ))
            idx += 1
    return trials
