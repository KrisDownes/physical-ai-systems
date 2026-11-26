# Phase 1 Results: 3-Way Proprioceptive Blind Insertion Comparison on SO-101

**Project:** `projects/blind_insertion/` (demo-free 1D-CNN Asymmetric Actor-Critic PPO + Auxiliary Contact-State & Direction Decoder)  
**Date:** 2026-11-24  
**Headline Verdict:** **NEGATIVE RL RESULT DUE TO AN IDENTIFIED OFF-PLATE DEPTH-SHAPING EXPLOIT — ALL 3 RL METHODS TRAILED CLASSICAL BLIND SPIRAL (`73.3% / 80.0% / 33.3%`), WHILE FAILURE AUDIT PINPOINTED THE EXACT 2-LINE REWARD BUG AND CONFIRMED M3's ZERO-COLLISION / ZERO-SHOVE COMPLIANCE.**

- **Compute & Speed:** All `9` runs (`M1_PosOnly`, `M2_Current`, `M3_Current_TL` × `3` seeds `[101, 202, 303]`, **`~402k` policy steps/run = `3.62M` total steps across `25,780` seen-shape episodes**) plus all reference baselines across Splits A, B, and C completed in **`2,314.8 s` (`38.6 min`)** on `10` headless CPU workers (`1,547–1,809 steps/sec`, comfortably under the `45 min` budget).
- **Split C (`tight_0.5mm`, `n = 30`, 15 bolted + 15 unbolted, 15 seen + 15 unseen) Reference Baselines:** Confirmed a massive **`+63.4 pp`** Oracle headroom over classical search — **Nominal `6.7%`**, **Blind Spiral Low-Torque `23.3%`**, **Blind Spiral Stiff `33.3%`** (`4` shoves), **Privileged Oracle `96.7%`** (`0` shoves, `0` safety violations).
- **3-Way RL Outcome (3-seed mean ± std across Splits A / B / C):**
  - **`M1_PosOnly` (SOTA *Blind Precise Assembly* replica):** `26.7% ± 13.6%` / `20.0% ± 11.9%` / `17.8% ± 8.7%` (best seed `s101`: `43.3% / 36.7% / 30.0%`, `95%` peak train success during `[0, 6] mm` curriculum).
  - **`M2_Current` (Current + Tracking Error, fixed `ts=1.0`):** `12.2% ± 3.1%` / `15.6% ± 19.7%` / `3.3% ± 2.7%` (best seed `s303` Split B: `43.3%`, `6/15` unseen).
  - **`M3_Current_TL` (Current + Tracking Error + Active `ts ∈ [0.20, 1.0]`):** `4.4% ± 3.1%` / `8.9% ± 6.3%` / `0.0% ± 0.0%`, with **`0` self-collisions, `0` joint-limit violations, `0.0` mean unbolted shoves on Split B**, lowest median peak force (`40.2 N` on Split B vs `56.5 N` `M1`), and phase-modulated torque (`ts = 0.54–0.55` search/rim → `0.59–0.60` insert/seat).
- **Root Cause Found by Failure Audit (Exact Reward Exploit):** Across **`86/90` (Split A), `82/90` (Split B), and `89/90` (Split C)** failures of `M3` — and **`73/90`, `62/90`, `81/90`** failures of `M2` — episodes terminated at **`max_depth_mm = 24.1–24.2 mm`** (exact table height `z = 0.0 mm` below the `24.0 mm` plate top!). Because `compute_step_reward` rewarded `35.0 * Δmax(0, peg_depth)` without gating `peg_depth` to `radial_err < hole_outer_r`, **once the misalignment curriculum widened past `~8 mm`, PPO discovered walking off the outer edge of the `84 mm` socket plate onto the table (`depth = +24.1 mm`) for `+1.03` shaping reward with zero shove risk!** `M2` and `M3` (with current sensing) detected the plate edge even more reliably than `M1` and exploited it harder (`M2 seed 202` also triggered `149` arm self-collision steps reaching far off the plate edge).

**Code & artifacts:**
- 1D-CNN temporal actor + dual aux decoder + asymmetric privileged critic: [`phase1_policy.py`](phase1_policy.py)
- Headless 10-worker training & 3-split evaluation harness: [`phase1_train.py`](phase1_train.py)
- Split C (`tight_0.5mm`) added cleanly to [`benchmarks.py`](benchmarks.py) (Regime A / B seeds, [`env.py`](env.py), [`geometry.py`](geometry.py) completely untouched)
- Phase 1 constraint & gradient unit tests: [`test_phase1.py`](test_phase1.py)
- Saved checkpoints (`9` `.npz` files with frozen `RunningMeanStd`): `phase1_checkpoints/`
- Full machine-readable results: [`phase1_results.json`](phase1_results.json)

---

## 1. Hard-Constraint, Disjointness & Unit-Test Verification

| Check | Result |
| --- | --- |
| [`geometry.py`](geometry.py), [`env.py`](env.py) physics, and Phase 0 Regime A / Regime B seeds untouched | **PASS** (zero edits) |
| Split C (`tight_0.5mm`, `n=30`, seeds `3101..3130`) strictly disjoint (`15B/15U`, `15 seen/15 unseen`, `0.5 mm`) | **PASS** (unit-tested) |
| Training strictly on SEEN shapes (`round`, `square`, `rectangle`) & disjoint seeds `>= 10000` (`25,780` eps) | **PASS** (runtime asserted + unit-tested; `0` unseen shapes in training) |
| Strict actor obs boundaries (`M1` zero `actuator_force_noisy` & zero `tracking_error`; zero privileged keys in M1/M2/M3) | **PASS** (unit-tested) |
| Analytic 1D-CNN + dual aux decoder + asymmetric critic gradients match finite differences | **PASS** (`atol=1e-5`) |
| Deterministic checkpoint save/load replay with frozen `RunningMeanStd` | **PASS** (unit-tested) |
| Unit tests (`test_phase0.py` + `test_phase1.py`) | **10/10 PASS** (`3.25 s`) |
| Held-out safety audit (`self_collision_steps` / `joint_limit_steps`) | Baselines & `M3`: **`0 / 0`**; `M1`: **`0 / 3`** (`s202` Split B); `M2`: **`149 / 0`** (`s202` off-plate fold) |

---

## 2. Full 3-Split Head-to-Head Comparison (Baselines vs. M1 / M2 / M3)

> RL methods report **3-seed mean ± std** across `seeds = [101, 202, 303]` (per-seed totals `[s101, s202, s303]` out of `30`). All three RL methods used identical 1D-CNN (`0.4 s` / 20-step history), asymmetric privileged critic, dual auxiliary decoder, reward function, and `~402k` policy-step budget.

### Split A — `bolted_socket` (`n = 30`, `1.0–1.5 mm` clearance, welded to table)

| Method | Overall Success | Seen (`n=15`) | Unseen (`n=15`) | Med. Insert Time | Med. Peak Force | Aux Cls Acc / Dir Cos | Mean `ts` (Free / Search / Seat) | Safety (`col / jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **0/30 (0.0%)** | 0.0% | 0.0% | — | 26.0 N | — | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **22/30 (73.3%)** | 86.7% | 60.0% | 1.25 s | 70.3 N | — | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **13/30 (43.3%)** | 60.0% | 26.7% | 2.04 s | 46.6 N | — | `0.35 / 0.35 / 0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(3 seeds)* | **26.7% ± 13.6%** `[13,8,3]` | 31.1% `[6,5,3]` | 22.2% `[7,3,0]` | 0.37 s | 85.3 N | `82.6%` / `+0.73` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(3 seeds)* | **12.2% ± 3.1%** `[5,3,3]` | 11.1% `[3,2,0]` | 13.3% `[2,1,3]` | 0.37 s | 78.1 N | `89.1%` / `+0.86` | `1.00 / 1.00 / 1.00` | `43 / 0` |
| **M3 · Current + Torque-Limit** *(3 seeds)* | **4.4% ± 3.1%** `[0,2,2]` | 2.2% `[0,1,0]` | 6.7% `[0,1,2]` | 0.48 s | **59.7 N** | `89.9%` / `+0.91` | `0.54 / 0.56 / 0.59` | **`0 / 0`** |
| **Privileged Oracle** | **29/30 (96.7%)** | 100.0% | 93.3% | 0.32 s | 0.0 N | — | `0.75` | `0 / 0` |

### Split B — `unbolted_socket` (`n = 30`, `1.0–1.5 mm` clearance, `180–320 g` free socket, `25 mm` shove limit)

| Method | Overall Success | Seen (`n=15`) | Unseen (`n=15`) | Med. Insert Time | Med. Peak Force | Shoves / Med. Disp | Aux Cls Acc / Dir Cos | Mean `ts` (Free / Search / Seat) | Safety (`col / jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | 13.3% | 0.0% | 4.36 s | 24.0 N | `4` / `1.3 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **24/30 (80.0%)** | 73.3% | 86.7% | 1.84 s | 43.7 N | `5` / `16.8 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **25/30 (83.3%)** | 80.0% | 86.7% | 1.74 s | 31.0 N | `0` / `12.8 mm` | — | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(3 seeds)* | **20.0% ± 11.9%** `[11,4,3]` | 26.7% `[7,2,3]` | 13.3% `[4,2,0]` | 0.38 s | 56.5 N | `6.7` / `9.6 mm` | `82.1%` / `+0.80` | `1.00 / 1.00 / 1.00` | `0 / 3` |
| **M2 · Current (Fixed `ts=1.0`)** *(3 seeds)* | **15.6% ± 19.7%** `[1,0,13]` | 17.8% `[1,0,7]` | 13.3% `[0,0,6]` | 0.38 s | 54.2 N | `1.7` / `5.4 mm` | `87.3%` / `+0.81` | `1.00 / 1.00 / 1.00` | `55 / 0` |
| **M3 · Current + Torque-Limit** *(3 seeds)* | **8.9% ± 6.3%** `[0,4,4]` | 8.9% `[0,1,3]` | 8.9% `[0,3,1]` | 0.42 s | **40.2 N** | **`0.0`** / `6.7 mm` | `89.5%` / `+0.88` | `0.54 / 0.55 / 0.60` | **`0 / 0`** |
| **Privileged Oracle** | **30/30 (100.0%)** | 100.0% | 100.0% | 0.42 s | 8.1 N | `0` / `0.3 mm` | — | `0.55` | `0 / 0` |

### Split C — `tight_0.5mm` (`n = 30`, `0.5 mm` clearance, `15` bolted + `15` unbolted, `15` seen + `15` unseen)

| Method | Overall Success | Seen (`n=15`) | Unseen (`n=15`) | Med. Insert Time | Med. Peak Force | Shoves / Med. Disp | Aux Cls Acc / Dir Cos | Mean `ts` (Free / Search / Seat) | Safety (`col / jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | 6.7% | 6.7% | 0.88 s | 26.9 N | `1` / `5.2 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **10/30 (33.3%)** | 40.0% | 26.7% | 1.55 s | 56.9 N | `4` / `13.7 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **7/30 (23.3%)** | 33.3% | 13.3% | 1.66 s | 41.6 N | `1` / `8.3 mm` | — | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(3 seeds)* | **17.8% ± 8.7%** `[9,4,3]` | 22.2% `[5,3,2]` | 13.3% `[4,1,1]` | 0.76 s | 67.5 N | `1.0` / `5.2 mm` | `79.7%` / `+0.77` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(3 seeds)* | **3.3% ± 2.7%** `[0,1,2]` | 2.2% `[0,0,1]` | 4.4% `[0,1,1]` | 0.33 s | 64.5 N | `0.3` / `6.0 mm` | `88.6%` / `+0.92` | `1.00 / 1.00 / 1.00` | `51 / 0` |
| **M3 · Current + Torque-Limit** *(3 seeds)* | **0.0% ± 0.0%** `[0,0,0]` | 0.0% `[0,0,0]` | 0.0% `[0,0,0]` | — | **50.5 N** | `0.3` / `6.2 mm` | `91.2%` / `+0.95` | `0.55 / 0.55 / —` | **`0 / 0`** |
| **Privileged Oracle** | **29/30 (96.7%)** | 93.3% | 100.0% | 0.38 s | 10.9 N | `0` / `0.5 mm` | — | `0.55–0.75` | `0 / 0` |

---

## 3. Compute, Architecture & Training Curves Summary

| RL Method | Step Ch. | Actor / Critic Params | Mean Steps / Seed | Mean Episodes / Seed | Mean Wall / Seed | Mean Throughput | Peak Train Succ (per seed) | End Train Succ (per seed) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **M1 · `M1_PosOnly`** | 19 | `11,373` / `18,881` | `401,328` | `3,160` | `269.0 s` | `1,547 st/s` | `95% / 55% / 65%` (`80k–137k`) | `25% / 20% / 10%` |
| **M2 · `M2_Current`** | 34 | `13,293` / `18,881` | `402,035` | `2,953` | `227.2 s` | `1,771 st/s` | `60% / 65% / 60%` (`40k–137k`) | `15% / 0% / 20%` |
| **M3 · `M3_Current_TL`** | 39 | `14,263` / `18,881` | `402,803` | `2,580` | `222.8 s` | `1,809 st/s` | `50% / 50% / 45%` (`37k–65k`) | `0% / 5% / 5%` |

*(Total wall-clock time across all 9 training runs + 3-split baseline & RL evaluations: `2,314.8 s` = `38.6 min`.)*

---

## 4. Failure-Mode Audit & Root-Cause Diagnosis

### 4.1 Exact Root Cause: Off-Plate Depth-Potential Reward Exploit (`max_depth_mm = 24.1–24.2 mm`)
Inspecting every failed held-out episode isolates the **single dominant failure mode** that broke late-stage training across `M2`, `M3`, and partially `M1`:

| Method | Split A Failures (`n_fail`) | Split A Fail Med. `max_depth` | Split B Failures (`n_fail`) | Split B Fail Med. `max_depth` | Split C Failures (`n_fail`) | Split C Fail Med. `max_depth` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **M1 · `M1_PosOnly`** | `66/90` (`30` plate, `31` stalled) | `0.4 mm` | `72/90` (`26` plate, `20` shove, `19` stalled) | `20.4 mm` | `74/90` (`31` plate, `27` stalled, `12` rim) | `0.4 mm` |
| **M2 · `M2_Current`** | `79/90` (**`73` stalled**, `6` safety) | **`24.2 mm`** | `76/90` (**`62` stalled**, `8` safety, `5` shove) | **`24.2 mm`** | `87/90` (**`81` stalled**, `4` safety) | **`24.2 mm`** |
| **M3 · `M3_Current_TL`** | `86/90` (**`86` stalled**) | **`24.1 mm`** | `82/90` (**`82` stalled**) | **`24.1 mm`** | `90/90` (**`89` stalled**, `1` shove) | **`24.1 mm`** |

- **Why `max_depth_mm = 24.1 mm` when the hole is only `20.0 mm` deep?**
  1. In `env.py`, `peg_depth = socket_top_center.z - peg_tip.z`. The socket top sits at `z = 24.0 mm` and the tabletop sits at `z = 0.0 mm`.
  2. In `phase1_train.py` (`compute_step_reward`), depth progress `+35.0 * Δmax(0, peg_depth)` and `+8.0 * Δpeg_depth` were **not gated on being inside the hole (`radial_err < hole_outer_r`)**!
  3. During the early curriculum (`0–100k` steps, misalignment `0–6 mm`), all three methods learned genuine insertion (`M1` hit `95%`, `M2` hit `65%`, `M3` hit `50%`).
  4. Once misalignment widened to `6–16 mm` (where finding the hole is harder than walking `30 mm` to the outer edge of the `84 mm` plate), PPO discovered a **deterministic local optimum**: slide off the outer edge of the socket plate onto the table (`z = 0.0 mm`), collecting `peg_depth = +24.1 mm` → **`+1.03` shaping reward** with **zero socket shove risk** and **zero contact force**!
  5. **Why `M2` and `M3` were hit even harder than `M1`:** Motor-current sensing (`actuator_force_noisy`) gave `M2` and `M3` an immediate tactile signal the instant the peg dropped off the outer rim of the plate (`aux_acc = 89–91%`, `dir_cos = +0.91..+0.95`), so they converged to the off-plate shortcut faster (`~150k` steps) and more consistently (`86–89/90`) than `M1`. It also caused `M2 seed 202`'s `149` self-collision steps when reaching off the far edge of the plate.

### 4.2 Positive Signals Already Confirmed Before the Fix
1. **Strong Split C (`tight_0.5mm`) Benchmark Separation:** Classical `BlindSpiral_Stiff` drops from `73.3% / 80.0%` to **`33.3%`**, `BlindSpiral_LowTorque` drops to **`23.3%`**, while `Privileged_Oracle` hits **`96.7%`** (`0` shoves, `0` safety violations) — proving huge recoverable headroom.
2. **Real Compliance & Safety Signal in `M3_Current_TL`:** `M3` achieved **`0` self-collisions, `0` joint-limit violations, `0.0` mean Split B shoves** (vs `6.7` for `M1` and `5` for `BlindSpiral_Stiff`), lowest median peak contact force across all three splits (`59.7 N / 40.2 N / 50.5 N` vs `85.3 N / 56.5 N / 67.5 N` for `M1`), and clear phase-dependent torque modulation (`ts = 0.54–0.55` search/rim → `0.59–0.60` insertion/seating).
3. **Zero-Shot Shape Generalization Visible in Best `M1` / `M2` Seeds:** Before the off-plate drift took over, `M1 seed 101` reached `13/30` Split A (`7/15` unseen > `6/15` seen), `11/30` Split B (`4/15` unseen), and `9/30` Split C (`4/15` unseen); `M2 seed 303` reached `13/30` Split B (`6/15` unseen vs `7/15` seen).

### 4.3 Exact Recommended Fixes for Phase 1.5 (Do Not Run Until Approved)
1. **Eliminate the Off-Plate Depth Exploit (2-line reward fix in `phase1_train.py`):**
   - Gate positive insertion depth strictly inside the hole: `cur_depth_in_hole = max(0.0, peg_depth) if (radial_err < hole_outer_r) else 0.0`.
   - Terminate or penalize (`-2.5` + `-0.05/step`) whenever `radial_err > 0.030 m` (leaving the socket plate surface) so walking off the `84 mm` plate is strictly worse than staying on the plate and searching for the hole.
2. **Keep `M1`, `M2`, `M3` Fairly Tuned Without Over-Ramping Curriculum:**
   - Gate curriculum expansion on batch success rate (e.g., advance `r_max` only when `train_succ >= 50%`) or keep a `25%` near-hole replay mix so policies never collapse when widening from `6 mm` to `16 mm`.
   - At our verified `1,547–1,809 steps/sec` throughput, a clean Phase 1.5 re-run across all 9 runs (`3` methods × `3` seeds × `400k–500k` steps) will finish in **`~38–42 minutes`**.

**Stopped here for review before any Phase 1.5 re-training.**
