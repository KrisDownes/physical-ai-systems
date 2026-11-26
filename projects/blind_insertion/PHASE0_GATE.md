# Phase 0 Gate Verification: Demo-Free Blind Proprioceptive Peg-in-Hole Insertion (SO-101)

**Project:** `projects/blind_insertion/` (clean standalone benchmark reusing Menagerie `robotstudio_so101` only — zero bottle-task code)  
**Date:** 2026-11-23  
**Gate Verdict:** **PASS — ALL PHASE 0 CRITERIA MET (`12/12` drop tests, `0/0` safety violations, Oracle `96.7%` bolted / `100.0%` unbolted, Nominal `0.0%` / `6.7%`, `2,154 steps/s` / `31.7 ep/s` headless, `6/6` unit tests). Stopped before Phase 1 RL training for review.**

**Code & artifacts:**
- Exact convex `mjfit` geometry & gravity drop-test gate: [`geometry.py`](geometry.py)
- 1 kHz / 50 Hz SO-101 STS3215 environment & strict observation boundary: [`env.py`](env.py)
- Deterministic Regime A (`bolted_socket`), Regime B (`unbolted_socket`), and `0.5 mm` tight suites: [`benchmarks.py`](benchmarks.py)
- Phase 0 reference controllers (Nominal, Blind Spiral `ts=1.0`, Blind Spiral `ts=0.35`, Privileged Oracle): [`controllers.py`](controllers.py)
- Phase 0 verification & throughput harness: [`phase0_verify.py`](phase0_verify.py)
- Unit tests (`unittest`): [`test_phase0.py`](test_phase0.py)
- Full machine-readable verification output: [`phase0_results.json`](phase0_results.json)

---

## 1. Phase 0 Gate Checklist & Safety Audit

| Gate Criterion | Target | Measured Result | Status |
| --- | --- | --- | --- |
| Exact convex geometry (`mjfit==0.1.0` on MuJoCo `4.0.1`, zero CoACD/V-HACD/SDF) | Installed & verified | Installed into `.venv-so101`; exact convex prism decomposition across all 6 shapes | **PASS** |
| Mandatory Gravity Drop-Test Gate (`6 shapes × {1.0 mm, 0.5 mm}`) | `100%` (`12/12`) | **`12/12` (`100.0%`)** — all fell to exact `20.00 mm` depth (`0.00 mm` drift, `0.00°` tilt) | **PASS** |
| Nominal Straight-Down (Regime A bolted / Regime B unbolted) | `< 10%` on both | **`0/30 (0.0%)`** bolted / **`2/30 (6.7%)`** unbolted | **PASS** |
| Privileged Oracle (Regime A bolted / Regime B unbolted) | `>= 95%` on both | **`29/30 (96.7%)`** bolted / **`30/30 (100.0%)`** unbolted (`6/6 = 100%` at `0.5 mm`) | **PASS** |
| Safety across all 258 Phase 0 benchmark rollouts (self-collision / joint-limit steps) | `0 / 0` | **`0 / 0`** | **PASS** |
| Strict `obs` vs `info["privileged"]` boundary + STS3215 quantization/backlash/torque clamping | Unit-tested | **`6/6` `unittest` PASS** (`2.52 s`) | **PASS** |
| Headless CPU throughput (`1 kHz` physics / `50 Hz` policy, `render=False`) | Fast training-ready | **`2,154.2 steps/s` (`31.68 ep/s`)** on 10 workers (`1,913.8 steps/s` on 8 workers) | **PASS** |

---

## 2. Exact Convex Geometry (`mjfit`) & Gravity Drop-Test Gate

`mjfit==0.1.0` was installed directly into `.venv-so101` (MuJoCo `4.0.1 >= 3.3`, `mapbox-earcut 2.1.0`, `numpy 2.5.4`). Every socket is decomposed into exact convex prism wedges around the hole with zero interior wall intrusion, and every peg is an exact inward offset of the hole outline (`plate_size = 84 mm`, `plate_thickness = 20 mm`, `bottom_thickness = 4 mm`, `peg_length = 36 mm`, target seated depth `>= 12 mm`).

Two strictly disjoint shape sets were generated:
- **Seen / Train shapes:** `round` (18 mm dia), `square` (17×17 mm), `rectangle` (20×14 mm)
- **Held-Out Unseen shapes (never used for tuning):** `hex` (20 mm circumdia), `triangle` (25 mm circumdia), `d_shape` (19 mm dia clipped flat at `x = +5.5 mm`, 20-vertex convex hull)

| Shape | Split | `1.0 mm` Clearance Depth / Drift / Tilt | `0.5 mm` Clearance Depth / Drift / Tilt | Gate |
| --- | --- | ---: | ---: | --- |
| `round` | Seen | `20.00 mm` / `0.00 mm` / `0.00°` | `20.00 mm` / `0.00 mm` / `0.00°` | **PASS** |
| `square` | Seen | `20.00 mm` / `0.00 mm` / `0.00°` | `20.00 mm` / `0.00 mm` / `0.00°` | **PASS** |
| `rectangle` | Seen | `20.00 mm` / `0.00 mm` / `0.00°` | `20.00 mm` / `0.00 mm` / `0.00°` | **PASS** |
| `hex` | Unseen | `20.00 mm` / `0.00 mm` / `0.00°` | `20.00 mm` / `0.00 mm` / `0.00°` | **PASS** |
| `triangle` | Unseen | `20.00 mm` / `0.00 mm` / `0.00°` | `20.00 mm` / `0.00 mm` / `0.00°` | **PASS** |
| `d_shape` | Unseen | `20.00 mm` / `0.00 mm` / `0.00°` | `20.00 mm` / `0.00 mm` / `0.00°` | **PASS** |

---

## 3. Physics, Control Rate & STS3215 Model Summary

- **Rates & Scene:** `1 kHz` MuJoCo physics (`timestep = 0.001 s`, `implicitfast`, elliptic cone, `solref=(0.005, 1.0)`, `solimp=(0.92, 0.98, 0.001, 0.5, 2.0)`), `50 Hz` policy (`20 substeps/step`, `DT = 0.02 s`), `render=False` headless-first. Scene XML is composed directly in memory from Menagerie `so101.xml` + `mjfit` convex prisms (zero temp files, zero lock contention).
- **5-DoF Kinematics & Virtual-Target Compliance:** Peg is mounted vertically inside the closed SO-101 parallel jaw (`22 mm` tip extension below jaw tips -> `2 mm` jaw clearance even at full `20 mm` hole bottom-out, with randomized per-episode in-hand grasp offset `[-1.5, +1.5] mm`). Policy actions `(dx, dy, dz)` (or joint deltas `delta_q`) map via damped 5-DoF nominal Jacobian IK anchored to `q_quantized` (preventing integrator windup into solid contact) while maintaining vertical orientation (`R = I`).
- **STS3215 Servo & Sensor Fidelity:**
  - 12-bit magnetic encoder quantization (`2π / 4096 ≈ 0.001534 rad`) on `q_quantized` and finite-difference `q_dot_quantized`.
  - Stateful mechanical backlash hysteresis deadband (`backlash_rad ∈ [0.003, 0.006] rad` per trial).
  - Noisy + 1-step (`20 ms`) lagged motor-current proxy (`actuator_force_noisy`: `±8%` multiplicative + `0.02 N·m` additive Gaussian).
  - Dynamic per-joint `torque_scale ∈ [0.15, 1.0]` directly clamping `model.actuator_forcerange[:5] = ±2.94 * torque_scale` (verified by `unittest` under saturated plate contact).
- **Strict Observation Boundary:**
  - **Actor `obs` (controller-legal only):** `q_quantized`, `q_dot_quantized`, `ee_pose_fk`, `ee_twist_fk`, `prev_action`, `tracking_error`, `actuator_force_noisy`, `torque_scale`, `time_s`. Zero socket pose, zero shape/clearance, zero true F/T wrench.
  - **Privileged `info["privileged"]` (training-only):** `socket_xy_error`, `peg_depth`, `contact_class ∈ {free, plate_contact, rim_contact, entering, seated}`, `peak_contact_force_N`, `socket_displacement_mm`, `socket_tilt_deg`, plus latent physics metadata.

---

## 4. Reference Controller Benchmarks

Both regimes contain **30 deterministic held-out trials** (`5` trials per shape: `15` seen + `15` unseen, clearances `1.0–1.5 mm`, stratified lateral misalignment `6–16 mm` with uniform direction, in-hand grasp offset `±1.5 mm`, peg-hole friction `μ ∈ [0.15, 0.45]`, backlash `∈ [0.003, 0.006] rad`, budget `250` steps = `5.0 s`).

### Regime A — `bolted_socket` (`n = 30`, seeds `1001..1030`, socket welded to table)

| Controller | Overall Success | Seen (`n=15`) | Unseen (`n=15`) | Median Insert Time | Median Peak Force | Failure Modes | Safety (`col / jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **0/30 (0.0%)** | 0/15 | 0/15 | — | 26.0 N | `14` rim-stalled, `12` rim-jam, `4` plate-jam | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **22/30 (73.3%)** | 13/15 | 9/15 | 1.25 s | 70.3 N | `5` plate-jam, `3` rim-jam | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **13/30 (43.3%)** | 9/15 | 4/15 | 2.04 s | 46.6 N | `10` plate-jam, `7` rim-jam | `0 / 0` |
| **Privileged Oracle** | **29/30 (96.7%)** | 15/15 | 14/15 | 0.32 s | 0.0 N | `1` hover-deadband (`regA-22-triangle`) | `0 / 0` |

*Per-shape successes on Regime A:*
- `BlindSpiral_Stiff_1.0`: `round 4/5`, `square 4/5`, `rectangle 5/5`, `hex 2/5`, `triangle 4/5`, `d_shape 3/5`
- `BlindSpiral_LowTorque_0.35`: `round 2/5`, `square 3/5`, `rectangle 4/5`, `hex 1/5`, `triangle 2/5`, `d_shape 1/5`
- `Privileged_Oracle`: `round 5/5`, `square 5/5`, `rectangle 5/5`, `hex 5/5`, `triangle 4/5`, `d_shape 5/5`

### Regime B — `unbolted_socket` (`n = 30`, seeds `2001..2030`, free socket `180–320 g`, table `μ ∈ [0.35, 0.55]`, shove limit `25 mm`)

| Controller | Overall Success | Seen (`n=15`) | Unseen (`n=15`) | Median Insert Time | Median Peak Force | Shove / Tip Failures | Median Socket Disp | Safety (`col / jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | 2/15 | 0/15 | 4.36 s | 24.0 N | `4` shove / `7` tip | 1.3 mm | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **24/30 (80.0%)** | 11/15 | 13/15 | 1.84 s | 43.7 N | **`5` shove** / `0` tip | **16.8 mm** | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **25/30 (83.3%)** | 12/15 | 13/15 | 1.74 s | **31.0 N** | **`0` shove** / `0` tip | **12.8 mm** | `0 / 0` |
| **Privileged Oracle** | **30/30 (100.0%)** | 15/15 | 15/15 | 0.42 s | 8.1 N | `0` shove / `0` tip | **0.3 mm** | `0 / 0` |

*Per-shape successes on Regime B:*
- `BlindSpiral_Stiff_1.0`: `round 3/5`, `square 4/5`, `rectangle 4/5`, `hex 4/5`, `triangle 5/5`, `d_shape 4/5` (`5` shoves, `1` plate-jam)
- `BlindSpiral_LowTorque_0.35`: `round 2/5`, `square 5/5`, `rectangle 5/5`, `hex 4/5`, `triangle 4/5`, `d_shape 5/5` (`0` shoves, `5` plate-jams)
- `Privileged_Oracle`: `5/5` across all 6 shapes (`30/30`)

### Tight-Clearance Sanity Suite (`0.5 mm` clearance, `n = 6`, all 6 shapes bolted)

| Controller | Success | Failure Modes | Safety (`col / jlim`) |
| --- | ---: | --- | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **0/6 (0.0%)** | `6` rim-stalled | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **1/6 (16.7%)** | `3` rim-jam, `2` plate-jam, `1` success | `0 / 0` |
| **Privileged Oracle** | **6/6 (100.0%)** | `6` success | `0 / 0` |

### Key Physical Takeaways for Phase 1
1. **Genuine cross-regime compliance trade-off:** Fixed high stiffness (`ts=1.0`) outperforms fixed low torque (`ts=0.35`) by **`+30.0 pp`** on bolted sockets (`73.3%` vs `43.3%`) because `ts=0.35` stalls on peg-socket friction (`17` jams vs `8`), whereas on unbolted sockets fixed `ts=1.0` triggers **`5` shove failures** (`> 25 mm`, median displacement `16.8 mm`, `43.7 N`) while `ts=0.35` eliminates shoves completely (`0` shoves, `12.8 mm`, `31.0 N`) at the cost of `5` friction jams. Neither fixed torque setting dominates both regimes — motivating active proprioceptive `torque_scale` modulation in Phase 1.
2. **Substantial headroom on unseen shapes, unbolted compliance, insertion speed, and tight `0.5 mm` clearance:** Classical blind spiral takes `1.25–1.84 s` (vs Oracle `0.32–0.42 s`), drops to `60.0%` (`9/15`) on unseen bolted shapes (`hex 2/5`, `d_shape 3/5`), and drops to `16.7%` (`1/6`) at `0.5 mm` clearance where Oracle hits `100.0%` (`6/6`).

---

## 5. Headless Multiprocessing Throughput Benchmark

Measured on 160 episodes (`10,880` 50 Hz policy steps = `217,600` 1 kHz physics substeps) with `render=False` and single-thread BLAS (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`):

| Configuration | Episodes | Policy Steps (50 Hz) | Physics Steps (1 kHz) | Wall Time | Policy `steps/sec` | `episodes/sec` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **8 CPU workers (headless)** | 160 | 10,880 | 217,600 | `5.68 s` | **`1,913.8 steps/s`** | **`28.14 ep/s`** |
| **10 CPU workers (headless)** | 160 | 10,880 | 217,600 | `5.05 s` | **`2,154.2 steps/s`** | **`31.68 ep/s`** |

At `~2,154 steps/s` (`31.7 ep/s`), `1,000,000` 50 Hz RL training steps run in **`~7.7 minutes`** wall-clock time on 10 CPU workers.

---

## 6. Unit Test Verification (`test_phase0.py`)

All **6/6** unit tests pass (`2.52 s` in `.venv-so101`):
- `test_exact_clearance_drop_tests_pass_100_percent` — verifies `12/12` drop tests across all 6 shapes at `1.0 mm` and `0.5 mm` (`depth >= 19.5 mm`).
- `test_zero_privileged_keys_in_obs_and_privileged_schema` — verifies `obs` contains strictly `ALLOWED_OBS_KEYS`, zero `FORBIDDEN_PRIVILEGED_KEYS`, exact `12-bit` quantization grid on `q_quantized`, and full `info["privileged"]` schema.
- `test_deterministic_seeding_identical_replay` — verifies bit-for-bit identical replay across two independent env instances on an unbolted `d_shape` trial.
- `test_torque_limit_scaling_clamps_actuator_force` — verifies per-joint `torque_scale` physically clamps `data.actuator_force` inside `[-2.94 * ts_j, +2.94 * ts_j]` under saturated contact.
- `test_backlash_deadband_holds_on_micro_reversal` — verifies stateful gear-lash deadband locks output during micro-reversals inside `±backlash_rad/2`.
- `test_nominal_fails_and_oracle_succeeds_bolted_and_unbolted` — verifies Nominal fails and Oracle succeeds with `0` self-collisions / `0` joint-limit violations on both bolted and unbolted `triangle` trials.

**Stopped here as requested — ready for review before Phase 1 RL training.**
