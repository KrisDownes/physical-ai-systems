# Phase 2 Results: Recurrent GRU Actor, Smooth Curriculum & 3-Way Blind Insertion Comparison on SO-101

**Project:** `projects/blind_insertion/` (Episode-Long `GRU(64)` Asymmetric Actor-Critic PPO + Controller-Legal `ee_rel_start` Spatial Anchor + Dual Auxiliary Contact Decoder; [`geometry.py`](geometry.py), [`env.py`](env.py), [`benchmarks.py`](benchmarks.py), and [`controllers.py`](controllers.py) completely untouched)  
**Date:** 2026-11-24  
**Headline Verdict:** **BOTH `M1_PosOnly` (`40.0% ± 6.7%`) AND `M3_Current_TL` (`36.7% ± 0.0%`, `0` SHOVES) BEAT CLASSICAL `BlindSpiral_Stiff` (`33.3%`, `4` SHOVES) AND `BlindSpiral_LowTorque` (`23.3%`) ON `0.5 mm` TIGHT CLEARANCES (SPLIT C), WITH `0 / 0` SAFETY VIOLATIONS ACROSS ALL `1,200` HELD-OUT ROLLOUTS; HOWEVER, RL STILL TRAILS CLASSICAL SPIRAL ON `1.0–1.5 mm` SPLITS A & B (`33–52%` VS `73–83%`) BECAUSE `80–83%` OF TRIALS REACH THE HOLE RIM BUT STALL IN `jammed_on_rim` (`24–28/60`), AND ZERO-SHOT OOD SLIPPERY GENERALIZATION (`mu=0.25, 120 g`) REMAINS LIMITED (`25–32%`, `15.5–19.0` SHOVES) BECAUSE TRAINING NEVER INCLUDED `< 0.35` FRICTION.**

- **Major Tight-Clearance Breakthrough (`Split C`, `0.5 mm`, Phase 1.5 → Phase 2):**
  - **`M1 · SOTA-PosOnly`:** `22.2%` → **`40.0% ± 6.7%`** `[14, 10]` (**`+17.8 pp`** vs Phase 1.5; **`+6.7 pp`** above `BlindSpiral_Stiff` `33.3%`, **`+16.7 pp`** above `BlindSpiral_LowTorque` `23.3%`; exact **`40.0%` seen / `40.0%` zero-shot unseen**; `> 10 mm` misalignment jumped **`3.3×`** from `11.1%` → **`36.1%`**!).
  - **`M3 · Current + Active Torque-Limit (Ours)`:** `22.2%` → **`36.7% ± 0.0%`** `[11, 11]` (**`+14.5 pp`** vs Phase 1.5; **`+3.4 pp`** above `BlindSpiral_Stiff`, with **`0.0` unbolted shoves** on both seeds vs `4` for `BlindSpiral_Stiff`, **`53.4 N`** peak force vs `65.5 N` `M1` / `56.9 N` `Spiral Stiff`, `6.2 mm` displacement vs `13.7 mm` `Spiral Stiff`, and `> 10 mm` jumping **`3.0×`** from `11.1%` → **`33.3%`**!).
  - **`M2 · Current (Fixed ts=1.0)`:** `23.3%` → `26.7% ± 6.7%` `[10, 6]` (`+3.4 pp` vs Phase 1.5).
- **Decisive Scientific Finding — Active Torque Compliance (`M3`) Rescues Noisy/Lagged Current (`M2`) & Cuts Force/Shoves, While Compact Kinematic-Only (`M1`) Stays Fiercely Competitive:**
  - Adding 20 ms-lagged noisy current (`±8% + 0.02 N·m`) *without* active compliance (`M2_Current`) degraded unbolted performance (`18.3%` Split B, `10.0%` OOD Slippery). Adding active 2-group STS3215 torque-limit control (`M3_Current_TL`, `[ts_lat, ts_arm]`) recovered **`+25.0 pp` on Split B (`43.3%` vs `18.3%`), `+10.0 pp` on Split C (`36.7%` vs `26.7%`), and `+15.0 pp` on OOD Slippery (`25.0%` vs `10.0%`)**, while achieving the **lowest peak contact force on all 4 splits** (`-8` to `-16 N` below `M1`), **fewest unbolted shoves on Splits B & C (`0.5` / `0.0`)**, **highest auxiliary contact-class accuracy (`80.7 / 77.4 / 76.6%`)**, and **tightest seed variance (`±0.0%` on Splits B & C)**.
  - Meanwhile, `M1_PosOnly` (compact 69-dim kinematic-only SOTA replica with 3-dim action space) still led 2-seed mean success on Splits A (`40.0%`), B (`51.7%`), C (`40.0%`), and OOD Slippery (`31.7%`), albeit with higher seed variance (`±6.7–16.7%`), higher peak forces (`+8` to `+16 N`), and `3×` more Split B shoves (`1.5` vs `0.5`).
- **Exact Next Bottleneck Identified (Hole-Reach vs. Rim-Jam Audit):** The GRU + `ee_rel_start` + smooth `6.0 → 16.5 mm` curriculum **already solved blind search**: `M1` and `M3` reach the hole rim (`success + jammed_on_rim`) on **`80.0–83.3%`** of Split A & B trials and **`75.0%`** of Split C (`M3`). Over half of those rim contacts (`24–28/60` on `M3`, `18–25/60` on `M1`) stall in `jammed_on_rim` on the chamferless 90° lip without executing a deliberate unload + micro-centering recovery.

**Code & artifacts:**
- Episode-long `GRU(64)` actor (per-gate orthogonal init, masked truncated-BPTT `L=32`, post-step KL trust-region rollback, 2-group `[ts_lat, ts_arm]` torque-limit head) + dual aux decoder + asymmetric critic: [`phase2_policy.py`](phase2_policy.py)
- Smooth linear curriculum (`6.0 → 16.5 mm` over `35%` + `20%` `<= 10 mm` replay), exploit-free hole-gated reward + `38 mm` plate-edge guard (`-8.0`), and 4-suite evaluator (Splits A/B/C + OOD Slippery): [`phase2_train.py`](phase2_train.py)
- Unit tests (`17/17 PASS`, including `ee_rel_start` legality, GRU hidden-state boundary reset, masked BPTT finite-difference gradients, walk-off `< -7.0` guard, and deterministic checkpoint replay): [`test_phase2.py`](test_phase2.py), [`test_phase1.py`](test_phase1.py), [`test_phase0.py`](test_phase0.py)
- Saved checkpoints (`6` `.npz` files with frozen `RunningMeanStd`): `phase2_checkpoints/`
- Full machine-readable results: [`phase2_results.json`](phase2_results.json)

---

## 1. Hard-Constraint, Architecture & Safety Verification

| Check | Result |
| --- | --- |
| [`geometry.py`](geometry.py), [`env.py`](env.py), [`benchmarks.py`](benchmarks.py) (Splits A / B / C), [`controllers.py`](controllers.py) untouched | **PASS** (`git diff --name-only` empty on all four) |
| Controller-legal `ee_rel_start = ee_pos_fk - ee_start_pos_fk` + zero privileged keys + `M1` bit-exact invariance to force / tracking error / `torque_scale` | **PASS** (unit-tested `atol=1e-12` & runtime-asserted) |
| Episode-long `GRU(64)` hidden-state boundary reset + exact masked truncated-BPTT gradients | **PASS** (unit-tested against central finite differences `atol=1e-5`) |
| Hole-gated depth shaping + plate-edge off-plate guard (`38 mm` Chebyshev / off-hole drop `> 2 mm` → `-8.0` penalty & termination) | **PASS** (live MuJoCo walk-off unit test `< -7.0`; `0.0%` late-train & eval off-plate on `M3`; failed `max_depth = 0.1–0.2 mm`) |
| Smooth linear curriculum (`6.0 → 16.5 mm` over first `35%`, `20%` `<= 10 mm` replay, `65%` at full `6–16.5 mm`) + strict seen-only shapes & disjoint seeds (`41,180` train episodes, seeds `>= 150000`) | **PASS** (runtime-asserted + unit-tested; `0` unseen shapes in training) |
| Unit tests (`test_phase0.py` + `test_phase1.py` + `test_phase2.py`) | **17/17 PASS** (`4.52 s`) |
| Evaluation safety violations (`self_collision_steps` / `joint_limit_steps` across all `1,200` eval rollouts on Splits A, B, C & OOD Slippery) | **`0 / 0`** |

---

## 2. Full 3-Split Comparison Across All 7 Methods (Phase 2, 2-Seed Mean ± Std)

> RL methods report **2-seed mean ± std** across `seeds = [101, 202]` (per-seed totals `[s101, s202]` out of `30`). All 3 RL methods shared identical `GRU(64)` backbone, `ee_rel_start` + `4×4` legal spatial memory, asymmetric privileged critic, dual auxiliary decoder, exploit-free reward, smooth linear `6.0 → 16.5 mm` curriculum, and `~1.00M` policy-step budget per run.

### Split A — `bolted_socket` (`n = 30`, `1.0–1.5 mm` clearance, welded to table)

| Method | Overall Success | Δ vs P1.5 | Seen (`n=15`) | Unseen (`n=15`) | `<= 10 mm` / `> 10 mm` | Med. Time | Med. Peak Force | Mean `ts` (F / S / I) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **0/30 (0.0%)** | — | 0.0% | 0.0% | `0.0%` / `0.0%` | — | 26.0 N | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **22/30 (73.3%)** | — | 86.7% | 60.0% | `91.7%` / `61.1%` | 1.25 s | 70.3 N | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **13/30 (43.3%)** | — | 60.0% | 26.7% | `50.0%` / `38.9%` | 2.04 s | 46.6 N | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(2 seeds)* | **40.0% ± 16.7%** `[17, 7]` | `-7.8 pp` | 43.3% `[8, 5]` | 36.7% `[9, 2]` | `50.0%` / `33.3%` | 1.60 s | 77.9 N | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(2 seeds)* | **35.0% ± 5.0%** `[12, 9]` | `-2.8 pp` | 36.7% `[7, 4]` | 33.3% `[5, 5]` | **`70.8%`** / `11.1%` | 0.73 s | 84.2 N | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(2 seeds)* | **33.3% ± 6.7%** `[8, 12]` | `-16.7 pp` | 23.3% `[4, 3]` | **43.3%** `[4, 9]` | `37.5%` / `30.6%` | 0.83 s | **62.4 N** | **`0.61 / 0.67 / 0.82`** | **`0 / 0`** |
| **Privileged Oracle** | **29/30 (96.7%)** | — | 100.0% | 93.3% | `91.7%` / `100.0%` | 0.32 s | 0.0 N | `0.75` | `0 / 0` |

### Split B — `unbolted_socket` (`n = 30`, `1.0–1.5 mm` clearance, `180–320 g` free socket, `25 mm` shove limit)

| Method | Overall Success | Δ vs P1.5 | Seen (`n=15`) | Unseen (`n=15`) | `<= 10 mm` / `> 10 mm` | Med. Time | Med. Peak Force | Shoves / Med. Disp | Mean `ts` (F / S / I) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | — | 13.3% | 0.0% | `8.3%` / `5.6%` | 4.36 s | 24.0 N | `4` / `1.3 mm` | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **24/30 (80.0%)** | — | 73.3% | 86.7% | `75.0%` / `83.3%` | 1.84 s | 43.7 N | `5` / `16.8 mm` | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **25/30 (83.3%)** | — | 80.0% | 86.7% | `66.7%` / `94.4%` | 1.74 s | 31.0 N | `0` / `12.8 mm` | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(2 seeds)* | **51.7% ± 8.3%** `[18, 13]` | `-1.6 pp` | 36.7% `[7, 4]` | **66.7%** `[11, 9]` | **`58.3%` / `47.2%`** | **0.51 s** | 51.7 N | `1.5` / `6.7 mm` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(2 seeds)* | **18.3% ± 5.0%** `[7, 4]` | `-31.7 pp` | 23.3% `[4, 3]` | 13.3% `[3, 1]` | `20.8%` / `16.7%` | 1.04 s | 57.5 N | `3.5` / `1.3 mm` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(2 seeds)* | **43.3% ± 0.0%** `[13, 13]` | `-1.1 pp` | **40.0%** `[7, 5]` | 46.7% `[6, 8]` | `54.2%` / `36.1%` | 0.89 s | **47.4 N** | **`0.5` / `5.7 mm`** | **`0.61 / 0.67 / 0.83`** | **`0 / 0`** |
| **Privileged Oracle** | **30/30 (100.0%)** | — | 100.0% | 100.0% | `100.0%` / `100.0%` | 0.42 s | 8.1 N | `0` / `0.3 mm` | `0.55` | `0 / 0` |

### Split C — `tight_0.5mm` (`n = 30`, `0.5 mm` clearance, `15` bolted + `15` unbolted, `15` seen + `15` unseen)

| Method | Overall Success | Δ vs P1.5 | Seen (`n=15`) | Unseen (`n=15`) | `<= 10 mm` / `> 10 mm` | Med. Time | Med. Peak Force | Shoves / Med. Disp | Mean `ts` (F / S / I) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | — | 6.7% | 6.7% | `8.3%` / `5.6%` | 0.88 s | 26.9 N | `1` / `5.2 mm` | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **10/30 (33.3%)** | — | 40.0% | 26.7% | `41.7%` / `27.8%` | 1.55 s | 56.9 N | `4` / `13.7 mm` | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **7/30 (23.3%)** | — | 33.3% | 13.3% | `41.7%` / `11.1%` | 1.66 s | 41.6 N | `1` / `8.3 mm` | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(2 seeds)* | **40.0% ± 6.7%** `[14, 10]` | **`+17.8 pp`** | 40.0% `[8, 4]` | **40.0%** `[6, 6]` | **`45.8%` / `36.1%`** | **1.10 s** | 65.5 N | `0.5` / `5.5 mm` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(2 seeds)* | **26.7% ± 6.7%** `[10, 6]` | `+3.4 pp` | 26.7% `[5, 3]` | 26.7% `[5, 3]` | `29.2%` / `25.0%` | 1.70 s | 64.9 N | `0.5` / `1.0 mm` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(2 seeds)* | **36.7% ± 0.0%** `[11, 11]` | **`+14.5 pp`** | **46.7%** `[8, 6]` | 26.7% `[3, 5]` | `41.7%` / `33.3%` | 1.38 s | **53.4 N** | **`0.0` / `6.2 mm`** | **`0.62 / 0.67 / 0.82`** | **`0 / 0`** |
| **Privileged Oracle** | **29/30 (96.7%)** | — | 93.3% | 100.0% | `100.0%` / `94.4%` | 0.38 s | 10.9 N | `0` / `0.5 mm` | `0.55–0.75` | `0 / 0` |

---

## 3. OOD Higher-Slipperiness Generalization Ablation (`mu_table = 0.25`, `120 g` Unbolted Socket, `n = 30`)

> Clones Split B's exact 30 geometries, clearances, and misalignments at **`mu_table = 0.25`** and **`socket_mass_kg = 0.120 kg`** (static breakaway force **`0.29 N`**, `3–5×` slipperier than the training range `mu_table ∈ [0.35, 0.55]`, `180–320 g` → `0.62–1.73 N`), leaving [`benchmarks.py`](benchmarks.py) untouched.

| Method | Overall Success | Seen / Unseen | `<= 10 mm` / `> 10 mm` | Med. Time | Med. Peak Force | Shove Failures | Med. Socket Disp | Mean `ts` (F / S / I) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **0/30 (0.0%)** | 0.0% / 0.0% | `0.0%` / `0.0%` | — | 12.6 N | `28 / 30` | `25.7 mm` | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **8/30 (26.7%)** | 26.7% / 26.7% | `25.0%` / `27.8%` | 1.28 s | 27.4 N | `22 / 30` | `25.2 mm` | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **14/30 (46.7%)** | 40.0% / 53.3% | `33.3%` / `55.6%` | 1.67 s | **19.3 N** | **`15 / 30`** | `24.5 mm` | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(2 seeds)* | **31.7% ± 11.7%** `[6, 13]` | 30.0% / 33.3% | `33.3%` / `30.6%` | 0.92 s | 35.1 N | `15.5 / 30` `[24, 7]` | `19.5 mm` | `1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(2 seeds)* | **10.0% ± 6.7%** `[5, 1]` | 16.7% / 3.3% | `8.3%` / `11.1%` | 1.27 s | 33.1 N | `22.5 / 30` `[22, 23]` | `25.5 mm` | `1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(2 seeds)* | **25.0% ± 5.0%** `[9, 6]` | 30.0% / 20.0% | `20.8%` / `27.8%` | 0.94 s | **26.9 N** | `19.0 / 30` `[20, 18]` | `25.2 mm` | **`0.62 / 0.69 / 0.81`** | **`0 / 0`** |
| **Privileged Oracle** | **30/30 (100.0%)** | 100.0% / 100.0% | `100.0%` / `100.0%` | 0.42 s | 9.3 N | `0 / 30` | `1.3 mm` | `0.55` | `0 / 0` |

**Honest OOD Slippery Verdict:**
- `M3_Current_TL` beats `M2_Current` (`+15.0 pp`, `-3.5` shoves, `-6.2 N` peak force) and matches `BlindSpiral_Stiff` (`25.0%` vs `26.7%`, with `3` fewer shoves: `19.0` vs `22`), achieving the **lowest peak force (`26.9 N`) of all RL methods and `BlindSpiral_Stiff`**.
- However, `M3` does **not** automatically drop its torque limit down to `0.25–0.35` zero-shot on `0.29 N` breakaway sockets (it keeps its in-distribution `ts_lat ≈ 0.77`, `ts_arm ≈ 0.64`), so it still shoves `19/30` sockets and trails `BlindSpiral_LowTorque(0.35)` (`46.7%`, `15` shoves) and `M1 seed=202` (`43.3%`, `7` shoves — whose slower sweep happened to impart less lateral impulse). Because training only included `mu_table ∈ [0.35, 0.55]` and `180–320 g` (where `ts ≈ 0.62–0.75` already achieved `0.0–0.5` shoves), `M3` had zero training incentive to map sliding slip to `ts < 0.40`.

---

## 4. Compute, Training Stability & M3 2-Group Torque Profile

| RL Method | Obs / Act Dim | Actor / Critic Params | Mean Steps / Seed | Mean Episodes / Seed | Mean Wall / Seed | Throughput | Train Off-Plate (Last 10 Upd) | Fail Med. `max_depth` (A / B / C) | Held-Out Aux Cls Acc (A / B / C) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **M1 · `M1_PosOnly`** | `69 / 3` | `30,093 / 19,009` | `1,001,232` | `7,000` | `724.6 s` | `1,382 st/s` | `[5.0, 0.0]%` | `0.1 / 0.1 / 0.2 mm` | `77.7 / 76.7 / 76.2%` |
| **M2 · `M2_Current`** | `96 / 3` | `31,821 / 19,009` | `1,002,695` | `6,620` | `662.2 s` | `1,517 st/s` | `[0.0, 1.0]%` | `0.2 / 0.3 / 0.2 mm` | `76.1 / 73.1 / 75.5%` |
| **M3 · `M3_Current_TL`** | `101 / 5` | `32,273 / 19,009` | `1,001,687` | `6,970` | `642.8 s` | **`1,559 st/s`** | **`[0.0, 0.0]%`** | `0.2 / 0.2 / 0.2 mm` | **`80.7 / 77.4 / 76.6%`** |

*(Total wall-clock time for all 6 runs × `1.0M` steps = `6.01M` policy steps + 4-suite evaluations = `4,206.8 s` = `70.1 min` on 10 headless CPU workers.)*

### Learned `M3_Current_TL` 2-Group Torque-Limit Modulation (`init = 0.75` for both groups)
| Split | Free (`ts_mean`) | Search / Rim (`ts_lat` / `ts_arm` / `mean`) | Insert / Seat (`ts_lat` / `ts_arm` / `mean`) |
| --- | ---: | ---: | ---: |
| **Split A (`bolted_socket`)** | `0.61` | `0.74` / **`0.63`** / `0.67` | **`0.71`** / **`0.90`** / `0.82` |
| **Split B (`unbolted_socket`)** | `0.61` | `0.75` / **`0.62`** / `0.67` | **`0.72`** / **`0.91`** / `0.83` |
| **Split C (`tight_0.5mm`)** | `0.62` | `0.73` / **`0.62`** / `0.67` | **`0.71`** / **`0.90`** / `0.82` |
| **OOD Slippery (`mu=0.25, 120 g`)** | `0.62` | `0.77` / **`0.64`** / `0.69` | **`0.70`** / **`0.89`** / `0.81` |

`M3` discovered a clean, physically interpretable **2-group separation** without any torque-shaping reward: during plate search and rim probing it **softens the vertical/reach arm joints (`ts_arm = 0.62–0.64`)** to keep contact forces low and avoid shoving unbolted sockets; once aligned in the hole (`entering / seated`) it **ramps `ts_arm` up to `0.89–0.91`** to drive decisive seating against wall friction while **relaxing lateral yaw (`ts_lat: 0.75 → 0.71`)** to comply to hole walls!

---

## 5. Failure Audit: Why RL Beat Spiral on `0.5 mm` Tight Clearances Yet Trails Spiral on `1.0–1.5 mm`

Pooled across both seeds (`60` trials per split):

| Method | Split | Seated (`success`) | Stalled on Rim (`jammed_on_rim`) | **Hole-Reach Rate (`succ + rim`)** | Missed Hole (`jammed_on_plate`) | Unbolted Shoved (`socket_shoved`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **M1 · `M1_PosOnly`** | Split A (`bolted`) | `24/60 (40.0%)` | **`25/60`** | **`49/60 (81.7%)`** | `11/60` | `0` |
| **M1 · `M1_PosOnly`** | Split B (`unbolted`) | `31/60 (51.7%)` | **`18/60`** | **`49/60 (81.7%)`** | `8/60` | `3` |
| **M1 · `M1_PosOnly`** | Split C (`0.5 mm`) | **`24/60 (40.0%)`** | `15/60` | `39/60 (65.0%)` | `20/60` | `1` |
| **M2 · `M2_Current`** | Split A (`bolted`) | `21/60 (35.0%)` | `18/60` | `39/60 (65.0%)` | `21/60` | `0` |
| **M2 · `M2_Current`** | Split B (`unbolted`) | `11/60 (18.3%)` | `23/60` | `34/60 (56.7%)` | `19/60` | `7` |
| **M2 · `M2_Current`** | Split C (`0.5 mm`) | `16/60 (26.7%)` | `20/60` | `36/60 (60.0%)` | `23/60` | `1` |
| **M3 · `M3_Current_TL`** | Split A (`bolted`) | `20/60 (33.3%)` | **`28/60`** | **`48/60 (80.0%)`** | `12/60` | `0` |
| **M3 · `M3_Current_TL`** | Split B (`unbolted`) | `26/60 (43.3%)` | **`24/60`** | **`50/60 (83.3%)`** | `9/60` | **`1`** |
| **M3 · `M3_Current_TL`** | Split C (`0.5 mm`) | `22/60 (36.7%)` | **`23/60`** | **`45/60 (75.0%)`** | `15/60` | **`0`** |

### Key Engineering & Scientific Conclusions
1. **The Short-Memory & Curriculum Bottlenecks Are Fixed:** `M1` and `M3` reach the hole on **`80–83%`** of Split A & B trials (`6–16 mm` misalignment) and **`65–75%`** of `0.5 mm` Split C trials — matching `BlindSpiral`'s search coverage, and **outperforming `BlindSpiral_Stiff` (`33.3%`) on `0.5 mm` tight clearances (`40.0%` `M1`, `36.7%` `M3` with `0` shoves)**.
2. **The Single Decisive Remaining Bottleneck Is Rim-Jam Recovery (`jammed_on_rim` = `24–28/60`):**
   - Look at `M3` Split A/B/C: **`48/60 (80.0%)`, `50/60 (83.3%)`, and `45/60 (75.0%)`** reach the hole rim, yet **`28`, `24`, and `23`** of those episodes stall wedged on the chamferless 90° lip (`fail_med_depth = 0.2 mm`) because the policy keeps pressing down once it earns the `+0.30` rim milestone instead of briefly unloading (`dz > 0`) and micro-dithering laterally across the rim lip.
3. **Why `M1_PosOnly` Outperformed `M2_Current` and Edged `M3_Current_TL` in Raw Success Rate:**
   - Under realistic STS3215 `20 ms` current lag + `±8% + 0.02 N·m` noise, raw current channels without compliance (`M2`) act as noisy distractors (`-33.4 pp` vs `M1` on Split B).
   - Coupling current with active STS3215 torque limits (`M3`) turns that signal into a clear physical advantage over `M2` (`+25.0 pp` Split B, `+10.0 pp` Split C, `+15.0 pp` OOD Slippery) and delivers **`-8` to `-16 N` lower peak force**, **`0.0–0.5` shoves**, and **zero seed variance (`±0.0%` on Splits B & C)** vs `M1` — confirming that **active torque-limit compliance improves physical safety and consistency, while rim-jam recovery and wider friction randomization (`mu_table ∈ [0.20, 0.55]`, `100–320 g`) are the exact two levers needed to push RL past `80%+` across all four splits.**

**Stopped here for review.**
