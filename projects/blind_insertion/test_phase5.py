"""Unit tests for Phase 5: Tactile-Amplifier AUROC, sigmoid spiral braking (`logit_g=+1.0`), 0.45mm clearance, replay."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.blind_insertion.env import InsertionTaskSpec
from projects.blind_insertion.geometry import SEEN_SHAPES
from projects.blind_insertion.phase1_train import verify_train_disjointness
from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
from projects.blind_insertion.phase4_policy import BASE_SPIRAL_ALPHA, P4_ALPHA_MAX, P4_ALPHA_MIN, Phase4ResidualBuilder
from projects.blind_insertion.phase4_train import PausableBaseSpiralController
from projects.blind_insertion.phase5_policy import P5_RES_FINE_XY_M, Phase5ResidualPPO
from projects.blind_insertion.phase5_tactile_snr import exact_auroc
from projects.blind_insertion.phase5_train import (
    MAX_WORKER_ENV_CACHE,
    P5_KEY_TO_IDX,
    LeanHeadlessGravCompEnv,
    _P5_LRU_ENV_CACHE,
    _clear_worker_cache_p5,
    _get_bounded_cached_env_p5,
    sample_training_batch_p5,
)


class Phase5BlindInsertionTest(unittest.TestCase):
    def test_exact_auroc_math(self):
        self.assertAlmostEqual(exact_auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 1.0, places=9)
        self.assertAlmostEqual(exact_auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 0.0, places=9)
        self.assertAlmostEqual(exact_auroc(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 0.5, places=9)

    def test_sigmoid_spiral_braking_fine_centering_and_0_45mm_geometry(self):
        b = Phase4ResidualBuilder('res_gravcomp_tl')
        pol = Phase5ResidualPPO(obs_dim=b.obs_dim, mode='res_gravcomp_tl', seed=11)

        # Zero-init (z=0): logit_g=+1.0 -> g_spiral = sigmoid(1.0) ~ 0.7311, alpha = 0.35
        _, d0, g0, ap0 = pol.compose_action(np.array([0.002, 0.0, -0.001]), np.zeros(6))
        self.assertAlmostEqual(g0, 1.0 / (1.0 + np.exp(-1.0)), places=6)
        self.assertEqual(ap0, (BASE_SPIRAL_ALPHA, BASE_SPIRAL_ALPHA))
        np.testing.assert_allclose(d0, g0 * np.array([0.002, 0.0, -0.001]), atol=1e-12)

        # Strong brake (z[3]=-1.4 -> logit=-2.5 -> g < 0.08) + fine micro-centering correction (0.5 mm)
        _, d_brk, g_brk, ap_lo = pol.compose_action(
            np.array([0.003, -0.002, -0.001]),
            np.array([0.33, -0.33, 0.0, -1.4, -1.0, -1.0]),
        )
        self.assertLess(g_brk, 0.08)
        self.assertEqual(ap_lo, (P4_ALPHA_MIN, P4_ALPHA_MIN))
        # Fine residual cap when braked stays close to P5_RES_FINE_XY_M (1.5 mm)
        self.assertLess(abs(d_brk[0]), 0.0010)
        self.assertGreater(P5_RES_FINE_XY_M, 0.001)

        # Confirm 0.45 mm tight clearance builds cleanly for all 3 seen shapes
        for sh in SEEN_SHAPES:
            env = GravCompBlindInsertionEnv(
                task=InsertionTaskSpec(trial_id=f't045-{sh}', seed=500001, shape=sh, clearance_m=0.00045, bolted=True),
                render=False,
            )
            try:
                obs, _ = env.reset()
                self.assertEqual(obs['q_quantized'].shape, (6,))
            finally:
                env.close()

    def test_phase5_tight_distribution_disjointness_and_checkpoint_replay(self):
        rng = np.random.default_rng(77)
        batch, cur = sample_training_batch_p5(rng, batch_size=400, seed_cursor=510000)
        self.assertEqual(cur, 510400)
        verify_train_disjointness(batch)
        tight_frac = sum(t.clearance_m <= 0.00075 for t in batch) / len(batch)
        self.assertGreater(tight_frac, 0.42)
        self.assertLess(tight_frac, 0.58)
        self.assertTrue(all(0.35 <= t.mu_table <= 0.55 and 0.180 <= t.socket_mass_kg <= 0.320 for t in batch))

        b = Phase4ResidualBuilder('res_gravcomp_tl')
        pol = Phase5ResidualPPO(obs_dim=b.obs_dim, mode='res_gravcomp_tl', seed=29)
        pol.W_mu[:] = rng.normal(0.0, 0.25, size=pol.W_mu.shape)
        pol.obs_rms.update(rng.normal(0.0, 1.0, size=(80, b.obs_dim)))

        task = InsertionTaskSpec(
            trial_id='p5-rep', seed=510999, shape='triangle', clearance_m=0.0005, bolted=False,
            socket_xy=(0.220, -0.004), misalign_xy=(-0.011, 0.008), mu_table=0.25, socket_mass_kg=0.120,
        )
        with tempfile.TemporaryDirectory() as tmp:
            p_ckpt = Path(tmp) / 'r2_v2.npz'
            pol.save(p_ckpt)
            loaded = Phase5ResidualPPO.load(p_ckpt)
            outs = []
            for p in (pol, loaded):
                env = GravCompBlindInsertionEnv(task=task, render=False)
                try:
                    obs, info = env.reset()
                    bc = PausableBaseSpiralController(torque_scale=1.0)
                    bc.reset(obs)
                    tb = Phase4ResidualBuilder('res_gravcomp_tl')
                    h = p.init_hidden()
                    trace = []
                    for _ in range(15):
                        bdxyz = bc.act_base(obs)
                        x = tb.step(obs, bdxyz, bc.phase, 0.0, bc.insert_stall_steps)
                        ea, d_xyz, g_sp, ap, _, h, c_p, _ = p.act_step_p4(x, h, bdxyz, deterministic=True)
                        bc.set_spiral_gate(g_sp)
                        tb.set_prev_controls(g_sp, ap)
                        self.assertTrue(0.0 < g_sp < 1.0)
                        self.assertTrue(P4_ALPHA_MIN <= ap[0] <= P4_ALPHA_MAX)
                        self.assertTrue(P4_ALPHA_MIN <= ap[1] <= P4_ALPHA_MAX)
                        obs, _, _, _, info = env.step(ea)
                        trace.append((d_xyz.tolist(), g_sp, ap, c_p, info['privileged']['peg_depth']))
                    outs.append(trace)
                finally:
                    env.close()
            self.assertEqual(outs[0], outs[1])

    def test_bounded_lru_env_cache_and_affinity_partitioning(self):
        # Verify 48 training keys partition evenly across 8 workers -> exactly 6 models/worker!
        self.assertEqual(len(P5_KEY_TO_IDX), 48)
        counts = [0] * 8
        for idx in P5_KEY_TO_IDX.values():
            counts[idx % 8] += 1
        self.assertEqual(counts, [6] * 8)

        # Verify bounded LRU cache strictly enforces max_cache cap and evicts cleanly
        _clear_worker_cache_p5()
        try:
            for k, c in enumerate((0.00045, 0.00050, 0.00060, 0.00070)):
                t = InsertionTaskSpec(trial_id=f'lru-{k}', seed=511000 + k, shape='round', clearance_m=c, bolted=True)
                _get_bounded_cached_env_p5(t, max_cache=2)
                self.assertLessEqual(len(_P5_LRU_ENV_CACHE), 2)
            self.assertEqual(len(_P5_LRU_ENV_CACHE), 2)
            self.assertEqual(MAX_WORKER_ENV_CACHE, 6)
        finally:
            _clear_worker_cache_p5()
            self.assertEqual(len(_P5_LRU_ENV_CACHE), 0)

    def test_lean_headless_env_is_bit_exact_and_20x_smaller(self):
        from projects.blind_insertion.phase3_gravcomp import GravCompBlindInsertionEnv
        from projects.blind_insertion.phase4_train import PausableBaseSpiralController

        task = InsertionTaskSpec(
            trial_id='lean-eq', seed=77, shape='square', clearance_m=0.0005,
            bolted=False, misalign_xy=(0.009, -0.005),
        )
        e_full = GravCompBlindInsertionEnv(task=task, render=False)
        e_lean = LeanHeadlessGravCompEnv(task=task, render=False)
        try:
            self.assertGreater(e_full.model.nbuffer / e_lean.model.nbuffer, 20.0)
            np.testing.assert_array_equal(e_full.model.body_mass, e_lean.model.body_mass)
            np.testing.assert_array_equal(e_full.model.body_inertia, e_lean.model.body_inertia)
            np.testing.assert_array_equal(e_full.model.body_ipos, e_lean.model.body_ipos)

            o1, _ = e_full.reset()
            o2, _ = e_lean.reset()
            c1 = PausableBaseSpiralController()
            c2 = PausableBaseSpiralController()
            c1.reset(o1)
            c2.reset(o2)
            for _ in range(120):
                o1, _, t1, tr1, i1 = e_full.step({'delta_xyz': c1.act_base(o1), 'gravcomp_alpha': 0.35})
                o2, _, t2, tr2, i2 = e_lean.step({'delta_xyz': c2.act_base(o2), 'gravcomp_alpha': 0.35})
                np.testing.assert_array_equal(e_full.data.qpos, e_lean.data.qpos)
                np.testing.assert_array_equal(e_full.data.actuator_force, e_lean.data.actuator_force)
                self.assertEqual(i1['episode_peak_contact_force_N'], i2['episode_peak_contact_force_N'])
                self.assertEqual(t1, t2)
                self.assertEqual(tr1, tr2)
                if t1 or tr1:
                    break
        finally:
            e_full.close()
            e_lean.close()


if __name__ == '__main__':
    unittest.main()
