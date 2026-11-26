"""Three deterministic held-out benchmark suites for Phase 0+ blind drawer opening.

  - Split A (`bolted_sticky`, n=30, seeds 1001..1030): Bolted cabinet, yaw in [-18, +18] deg,
    high rail friction [2.5, 5.5] N + sticky detent F_detent in [5.0, 9.0] N at d_detent in [10, 40] mm.
  - Split B (`unbolted_cabinet`, n=30, seeds 2001..2030): Unbolted cabinet (mass 400-750 g,
    mu_table 0.40-0.65), yaw in [-18, +18] deg, moderate detent [3.0, 7.0] N.
  - Split OOD (`ood_light_height_angle`, n=20, seeds 3001..3020): Strictly OOD lighter unbolted
    cabinet (280-350 g, mu_table = 0.35), unseen height shift +-15 mm, steeper yaw +-22 deg.
"""
from __future__ import annotations

import numpy as np

from .env import DrawerTaskSpec

SPLIT_A_SEED_OFFSET = 1001
SPLIT_B_SEED_OFFSET = 2001
SPLIT_OOD_SEED_OFFSET = 3001
TRAIN_SEED_OFFSET = 10000


def build_split_a_bolted_sticky() -> list[DrawerTaskSpec]:
    """Split A: Bolted cabinet, yaw +-18 deg, high rail friction + sticky detent (5-9 N)."""
    rng = np.random.default_rng(SPLIT_A_SEED_OFFSET)
    yaws = np.linspace(-18.0, 18.0, 30)
    rng.shuffle(yaws)
    trials: list[DrawerTaskSpec] = []
    for i in range(30):
        trials.append(DrawerTaskSpec(
            trial_id=f'drw-a-{i:02d}',
            seed=SPLIT_A_SEED_OFFSET + i,
            bolted=True,
            cabinet_xy=(float(0.336 + rng.uniform(-0.008, 0.008)), float(rng.uniform(-0.012, 0.012))),
            height_shift_m=0.0,
            rail_yaw_deg=float(yaws[i]),
            f_rail_N=float(rng.uniform(2.5, 5.5)),
            rail_damping=float(rng.uniform(10.0, 18.0)),
            rail_bind_mu=1.80,
            detent_travel_mm=float(rng.uniform(10.0, 40.0)),
            detent_force_N=float(rng.uniform(5.0, 9.0)),
            detent_width_mm=6.0,
            mu_jaw_handle=0.85,
            mu_table=0.55,
            cabinet_mass_kg=0.600,
            backlash_rad=0.0045,
            max_steps=300,
        ))
    return trials


def build_split_b_unbolted_cabinet() -> list[DrawerTaskSpec]:
    """Split B: Unbolted cabinet (400-750 g, mu_table 0.40-0.65), yaw +-18 deg, detent 3-7 N."""
    rng = np.random.default_rng(SPLIT_B_SEED_OFFSET)
    yaws = np.linspace(-18.0, 18.0, 30)
    rng.shuffle(yaws)
    trials: list[DrawerTaskSpec] = []
    for i in range(30):
        trials.append(DrawerTaskSpec(
            trial_id=f'drw-b-{i:02d}',
            seed=SPLIT_B_SEED_OFFSET + i,
            bolted=False,
            cabinet_xy=(float(0.336 + rng.uniform(-0.008, 0.008)), float(rng.uniform(-0.012, 0.012))),
            height_shift_m=0.0,
            rail_yaw_deg=float(yaws[i]),
            f_rail_N=float(rng.uniform(0.8, 1.8)),
            rail_damping=float(rng.uniform(8.0, 12.0)),
            rail_bind_mu=1.80,
            detent_travel_mm=float(rng.uniform(10.0, 40.0)),
            detent_force_N=float(rng.uniform(3.0, 7.0)),
            detent_width_mm=2.5,
            mu_jaw_handle=0.85,
            mu_table=float(rng.uniform(0.40, 0.65)),
            cabinet_mass_kg=float(rng.uniform(0.400, 0.750)),
            backlash_rad=0.0045,
            max_steps=300,
        ))
    return trials


def build_split_ood_light_height_angle() -> list[DrawerTaskSpec]:
    """Split OOD (n=20): Light unbolted (280-350 g, mu=0.35), height shifted +-15 mm, steep yaw +-22 deg."""
    rng = np.random.default_rng(SPLIT_OOD_SEED_OFFSET)
    yaws = np.concatenate([np.linspace(-22.0, -14.0, 10), np.linspace(14.0, 22.0, 10)])
    rng.shuffle(yaws)
    trials: list[DrawerTaskSpec] = []
    for i in range(20):
        dz = 0.015 if (i % 2 == 0) else -0.015
        trials.append(DrawerTaskSpec(
            trial_id=f'drw-ood-{i:02d}',
            seed=SPLIT_OOD_SEED_OFFSET + i,
            bolted=False,
            cabinet_xy=(float(0.336 + rng.uniform(-0.006, 0.006)), float(rng.uniform(-0.010, 0.010))),
            height_shift_m=dz,
            rail_yaw_deg=float(yaws[i]),
            f_rail_N=float(rng.uniform(0.6, 1.1)),
            rail_damping=float(rng.uniform(7.0, 10.0)),
            rail_bind_mu=1.80,
            detent_travel_mm=float(rng.uniform(10.0, 40.0)),
            detent_force_N=float(rng.uniform(2.5, 4.5)),
            detent_width_mm=2.5,
            mu_jaw_handle=0.85,
            mu_table=0.35,
            cabinet_mass_kg=float(rng.uniform(0.280, 0.350)),
            backlash_rad=0.0045,
            max_steps=300,
        ))
    return trials
