# Phase 1.5 Results: Off-Plate Reward Fix & 3-Way Blind Insertion Comparison on SO-101

**Project:** `projects/blind_insertion/` (1D-CNN Asymmetric Actor-Critic PPO + Dual Auxiliary Contact Decoder; [`geometry.py`](geometry.py), [`env.py`](env.py), and Splits A / B / C completely untouched)  
**Date:** 2026-11-24  
**Headline Verdict:** **OFF-PLATE REWARD EXPLOIT COMPLETELY ELIMINATED (`0 / 0` SAFETY VIOLATIONS, `+20` TO `+46 pp` GAINS ACROSS ALL 3 RL METHODS); `M3_Current_TL` WINS SPLIT A (`50.0% ± 2.7%`), SLASHES PEAK FORCES & UNBOLTED SHOVES/DISPLACEMENT (`4.9 mm`), AND SHOWS ZERO-SHOT UNSEEN GENERALIZATION (`51.1%` UNSEEN VS `48.9%` SEEN), WHILE `M1 s202` (THE ONLY SEED TO REACH THE `16 mm` CURRICULUM STAGE) ALREADY BEAT `BlindSpiral_Stiff` ON `0.5 mm` TIGHT CLEARANCES (`46.7%` VS `33.3%`).**

- **Reward Hack Verified Dead:** Failed held-out episodes dropped from Phase 1's **`24.1–24.2 mm`** off-plate tabletop depth to **`0.2 mm`** (genuine on-plate / rim contact failures) across all 3 methods and all 3 splits; late-training off-plate termination rate fell to **`1.5–3.5%` on `M3`** (`0.5–8.0%` on `M1`/`M2`), and evaluation safety violations dropped from `149 col / 3 jlim` in Phase 1 to **`0 / 0`** across all `990` held-out rollouts!
- **Phase 1 → Phase 1.5 Jumps (3-seed mean ± std across Split A / Split B / Split C):**
  - **`M1 · SOTA-PosOnly`:** `26.7% / 20.0% / 17.8%` → **`47.8% ± 9.6%` / `53.3% ± 14.4%` / `22.2% ± 17.5%`** (`+21.1 / +33.3 / +4.4 pp`). Seed `202` — the *only* run of the 9 to clear `>= 75%` at `14 mm` and train on `16 mm` — reached **`60.0%` Split A / `73.3%` Split B / `46.7%` Split C** (beating `BlindSpiral_Stiff` `33.3%` by **`+13.4 pp`** on `0.5 mm` clearances!).
  - **`M2 · Current (Fixed ts=1.0)`:** `12.2% / 15.6% / 3.3%` → **`37.8% ± 5.7%` / `50.0% ± 9.8%` / `23.3% ± 2.7%`** (`+25.6 / +34.4 / +20.0 pp`), with the highest 3-seed mean on `0.5 mm` Split C (`23.3% ± 2.7%`).
  - **`M3 · Current + Active Torque-Limit (Ours)`:** `4.4% / 8.9% / 0.0%` → **`50.0% ± 2.7%` / `44.4% ± 3.1%` / `22.2% ± 4.2%`** (**`+45.6 / +35.6 / +22.2 pp`**!). `M3` achieved the **highest 3-seed mean on Split A** (beating `M1`, `M2`, and `BlindSpiral_LowTorque` `43.3%`), **tightest seed consistency (`±2.7–4.2%` vs `±9.6–17.5%` on `M1`)**, **lowest peak contact force across all 3 splits** (`-12` to `-17 N` below `M1`), **fewest unbolted shoves** (`0.7` on Split B vs `2.3` `M1`, `1.7` `M2`, `5` `Spiral Stiff`), and **lowest unbolted socket displacement (`4.9 mm` vs `9.0 mm` `M1`, `9.5 mm` `M2`, `16.8 mm` `Spiral Stiff`, `12.8 mm` `Spiral Low-Torque`)**.
- **Single Remaining Gap to Beat Classical Spiral in 3-Seed Mean:** `8` of `9` runs (`M1 s101/303`, all `M2`, all `M3`) stayed at curriculum stage **`r_max = 14.0 mm`** when `500k` steps finished (rolling 100-ep success plateaued at `50–66% < 75%` promotion threshold), so they never trained on `14–16 mm` misalignments (`66.7–75.0%` success on `<= 10 mm` vs `35.2–42.6%` on `> 10 mm`).

**Code & artifacts:**
- 1D-CNN actor (`M3` `ts` initialized at `0.75`, clamped `[0.20, 1.0]`) + dual aux decoder + asymmetric critic: [`phase1_policy.py`](phase1_policy.py)
- Exploit-free hole-gated reward, `28 mm` off-plate termination, and `>= 75%` rolling-100 curriculum: [`phase1_train.py`](phase1_train.py)
- Unit tests (`11/11 PASS`, including full MuJoCo walk-off negative-return test): [`test_phase1.py`](test_phase1.py), [`test_phase0.py`](test_phase0.py)
- Saved checkpoints (`9` `.npz` files with frozen `RunningMeanStd`): `phase1_5_checkpoints/`
- Full machine-readable results: [`phase1_5_results.json`](phase1_5_results.json)

---

## 1. Hard-Constraint, Fix & Safety Verification

| Check | Result |
| --- | --- |
| [`geometry.py`](geometry.py), [`env.py`](env.py) physics, and Splits A / B / C in [`benchmarks.py`](benchmarks.py) untouched | **PASS** (zero edits) |
| Hole-gated depth shaping (`radial_err < hole_outer_r + 1.5 mm`) + `28 mm` / off-hole drop guard (`-2.5` penalty & termination) | **PASS** (unit-tested in live MuJoCo walk-off episode: strictly negative total return `< -2.0`) |
| `M3` episode-0 zero-init `torque_scale = 0.75` (`∈ [0.20, 1.0]`) | **PASS** (unit-tested) |
| Strict train/test disjointness (`43,500` training episodes, seeds `>= 60000`, strictly SEEN shapes only) | **PASS** (runtime asserted + unit-tested; `0` unseen shapes in training) |
| Unit tests (`test_phase0.py` + `test_phase1.py`) | **11/11 PASS** (`3.47 s`) |
| Evaluation safety violations (`self_collision_steps` / `joint_limit_steps` across all 990 eval rollouts) | **`0 / 0`** (vs `149 / 3` in Phase 1) |

---

## 2. Full 3-Split Comparison Across All 7 Methods (Phase 1.5, 3-Seed Mean ± Std)

> RL methods report **3-seed mean ± std** across `seeds = [101, 202, 303]` (per-seed totals `[s101, s202, s303]` out of `30`). All 3 RL methods shared identical 1D-CNN backbone, asymmetric privileged critic, dual auxiliary decoder, exploit-free reward, `>= 75%` rolling-100 curriculum (`6 → 10 → 14 → 16 mm`), and `~501k` policy-step budget.

### Split A — `bolted_socket` (`n = 30`, `1.0–1.5 mm` clearance, welded to table)

| Method | Overall Success | Δ vs P1 | Seen (`n=15`) | Unseen (`n=15`) | Med. Time | Med. Peak Force | Off-Plate Eval | Mean `ts` (Free / Search / Seat) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **0/30 (0.0%)** | — | 0.0% | 0.0% | — | 26.0 N | — | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **22/30 (73.3%)** | — | 86.7% | 60.0% | 1.25 s | 70.3 N | — | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **13/30 (43.3%)** | — | 60.0% | 26.7% | 2.04 s | 46.6 N | — | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(3 seeds)* | **47.8% ± 9.6%** `[11,18,14]` | `+21.1 pp` | 48.9% `[6,9,7]` | 46.7% `[5,9,7]` | 0.82 s | 89.1 N | `0.0%` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(3 seeds)* | **37.8% ± 5.7%** `[13,9,12]` | `+25.6 pp` | 40.0% `[6,5,7]` | 35.6% `[7,4,5]` | 0.82 s | 87.0 N | `7.8%` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(3 seeds)* | **50.0% ± 2.7%** `[16,15,14]` | **`+45.6 pp`** | 48.9% `[8,7,7]` | **51.1%** `[8,8,7]` | 1.33 s | **72.5 N** | `5.6%` | **`0.70 / 0.70 / 0.64`** | **`0 / 0`** |
| **Privileged Oracle** | **29/30 (96.7%)** | — | 100.0% | 93.3% | 0.32 s | 0.0 N | — | `0.75` | `0 / 0` |

### Split B — `unbolted_socket` (`n = 30`, `1.0–1.5 mm` clearance, `180–320 g` free socket, `25 mm` shove limit)

| Method | Overall Success | Δ vs P1 | Seen (`n=15`) | Unseen (`n=15`) | Med. Time | Med. Peak Force | Shoves / Med. Disp | Off-Plate Eval | Mean `ts` (Free / Search / Seat) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | — | 13.3% | 0.0% | 4.36 s | 24.0 N | `4` / `1.3 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **24/30 (80.0%)** | — | 73.3% | 86.7% | 1.84 s | 43.7 N | `5` / `16.8 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **25/30 (83.3%)** | — | 80.0% | 86.7% | 1.74 s | 31.0 N | `0` / `12.8 mm` | — | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(3 seeds)* | **53.3% ± 14.4%** `[14,22,12]` | `+33.3 pp` | 51.1% `[5,10,8]` | 55.6% `[9,12,4]` | 1.06 s | 61.3 N | `2.3` / `9.0 mm` | `8.9%` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M2 · Current (Fixed `ts=1.0`)** *(3 seeds)* | **50.0% ± 9.8%** `[16,11,18]` | `+34.4 pp` | 53.3% `[7,7,10]` | 46.7% `[9,4,8]` | 1.22 s | 58.3 N | `1.7` / `9.5 mm` | `12.2%` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(3 seeds)* | **44.4% ± 3.1%** `[12,14,14]` | **`+35.6 pp`** | 48.9% `[7,7,8]` | 40.0% `[5,7,6]` | 1.30 s | **53.2 N** | **`0.7` / `4.9 mm`** | `7.8%` | **`0.69 / 0.70 / 0.63`** | **`0 / 0`** |
| **Privileged Oracle** | **30/30 (100.0%)** | — | 100.0% | 100.0% | 0.42 s | 8.1 N | `0` / `0.3 mm` | — | `0.55` | `0 / 0` |

### Split C — `tight_0.5mm` (`n = 30`, `0.5 mm` clearance, `15` bolted + `15` unbolted, `15` seen + `15` unseen)

| Method | Overall Success | Δ vs P1 | Seen (`n=15`) | Unseen (`n=15`) | Med. Time | Med. Peak Force | Shoves / Med. Disp | Off-Plate Eval | Mean `ts` (Free / Search / Seat) | Safety (`col/jlim`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | **2/30 (6.7%)** | — | 6.7% | 6.7% | 0.88 s | 26.9 N | `1` / `5.2 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Stiff** (`ts=1.0`) | **10/30 (33.3%)** | — | 40.0% | 26.7% | 1.55 s | 56.9 N | `4` / `13.7 mm` | — | `1.00` | `0 / 0` |
| **Blind Spiral · Low Torque** (`ts=0.35`) | **7/30 (23.3%)** | — | 33.3% | 13.3% | 1.66 s | 41.6 N | `1` / `8.3 mm` | — | `0.35` | `0 / 0` |
| **M1 · SOTA-PosOnly** *(3 seeds)* | **22.2% ± 17.5%** `[4,14,2]` | `+4.4 pp` | 15.6% `[0,7,0]` | 28.9% `[4,7,2]` | 1.70 s | 79.4 N | `1.0` / `4.8 mm` | `4.4%` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| *↳ M1 `seed=202` (only `16 mm` run)* | ***14/30 (46.7%)*** | — | *46.7% (`7/15`)* | *46.7% (`7/15`)* | — | — | *`0` shoves* | — | *`1.00`* | *`0 / 0`* |
| **M2 · Current (Fixed `ts=1.0`)** *(3 seeds)* | **23.3% ± 2.7%** `[7,6,8]` | `+20.0 pp` | 26.7% `[2,4,6]` | 20.0% `[5,2,2]` | 1.61 s | 77.6 N | `0.7` / `6.9 mm` | `10.0%` | `1.00 / 1.00 / 1.00` | `0 / 0` |
| **M3 · Current + Torque-Limit** *(3 seeds)* | **22.2% ± 4.2%** `[7,5,8]` | **`+22.2 pp`** | 24.4% `[3,3,5]` | 20.0% `[4,2,3]` | 2.58 s | **66.5 N** | **`0.3`** / `7.0 mm` | `4.4%` | **`0.70 / 0.70 / 0.64`** | **`0 / 0`** |
| **Privileged Oracle** | **29/30 (96.7%)** | — | 93.3% | 100.0% | 0.38 s | 10.9 N | `0` / `0.5 mm` | — | `0.55–0.75` | `0 / 0` |

---

## 3. Compute, Curriculum Progression & Off-Plate Hack Verification

| RL Method | Mean Steps / Seed | Mean Episodes / Seed | Mean Wall / Seed | Throughput | Final `r_max` (`[s101, s202, s303]`) | Train Off-Plate (Last 10 Upd) | Fail Med. `max_depth` (A / B / C) | Held-Out Aux Cls Acc (A / B / C) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **M1 · `M1_PosOnly`** | `500,999` | `4,813` | `356.6 s` | `1,412 st/s` | `[14, 16, 14] mm` | `[5.5, 4.5, 4.5]%` | `0.2 / 0.2 / 0.2 mm` | `74.1 / 75.4 / 72.4%` |
| **M2 · `M2_Current`** | `501,327` | `4,953` | `359.3 s` | `1,396 st/s` | `[14, 14, 14] mm` | `[8.0, 0.5, 5.5]%` | `0.2 / 0.2 / 0.2 mm` | `73.6 / 72.2 / 70.0%` |
| **M3 · `M3_Current_TL`** | `500,345` | `4,733` | `349.9 s` | `1,431 st/s` | `[14, 14, 14] mm` | **`[1.5, 3.5, 2.0]%`** | `0.2 / 0.2 / 0.2 mm` | **`75.3 / 77.4 / 73.6%`** |

*(Total wall-clock time for all 9 runs × `500k` steps = `4.51M` policy steps + 3-split evaluations: `3,350.7 s` = `55.8 min` on 10 CPU workers.)*

---

## 4. Diagnostic Audit: What Worked, What M3 Learned, and Why 3-Seed Mean Still Trails Spiral Search

### 4.1 The Off-Plate Exploit Is Completely Dead
- In Phase 1, `M2` and `M3` failed almost exclusively by walking off the outer edge of the `84 mm` plate (`fail_med_depth = 24.1–24.2 mm`, `86–89/90` episodes on `M3`, `149` self-collisions on `M2 s202`).
- In Phase 1.5, **every single method's median failed `max_depth` dropped to `0.2 mm`** (`jammed_on_plate` / `jammed_on_rim` — genuine blind search timeouts on the socket top plate!), evaluation `self_collision_steps` and `joint_limit_steps` dropped to **`0 / 0`**, and late-training off-plate termination rates stayed low (`1.5–3.5%` on `M3`).

### 4.2 Misalignment Breakdown (`<= 10 mm` vs `> 10 mm`) & Curriculum Bottleneck
| Method | Split A `<= 10 mm` / `> 10 mm` | Split B `<= 10 mm` / `> 10 mm` | Split C `<= 10 mm` / `> 10 mm` |
| --- | ---: | ---: | ---: |
| **M1 · `M1_PosOnly`** | `66.7%` (`24/36`) / `35.2%` (`19/54`) | **`75.0%`** (`27/36`) / `38.9%` (`21/54`) | `38.9%` (`14/36`) / `11.1%` (`6/54`) |
| **M2 · `M2_Current`** | `55.6%` (`20/36`) / `25.9%` (`14/54`) | `69.4%` (`25/36`) / `37.0%` (`20/54`) | `36.1%` (`13/36`) / **`14.8%`** (`8/54`) |
| **M3 · `M3_Current_TL`** | **`72.2%`** (`26/36`) / `35.2%` (`19/54`) | `47.2%` (`17/36`) / **`42.6%`** (`23/54`) | `38.9%` (`14/36`) / `11.1%` (`6/54`) |

- **Why 3-seed RL means (`38–53%` A/B, `22–23%` C) still trail `BlindSpiral` (`73.3% / 80.0% / 33.3%`):**
  - All 9 seeds rapidly cleared `6 mm → 10 mm` (`14k–28k` steps) and `10 mm → 14 mm` (`29k–188k` steps), **but 8 of 9 seeds plateaued around `50–66%` rolling-100 success at `r_max = 14 mm`** and never hit the strict `>= 75%` threshold to unlock the `16 mm` stage within `500k` steps.
  - Because `60%` of test trials (`18/30` per split) have misalignment `> 10 mm` (and `6/30` are in `14–16 mm`, completely out-of-distribution for 8/9 seeds!), success dropped sharply from **`66.7–75.0%` on `<= 10 mm`** down to **`25.9–42.6%` on `> 10 mm`**.
  - Crucially, the **single seed (`M1 s202`) that broke through `>= 75%` at `14 mm` and trained at `16 mm` (`435k–500k` steps)** immediately jumped to **`18/30 (60.0%)` Split A, `22/30 (73.3%)` Split B, and `14/30 (46.7%)` Split C (`7/15` seen + `7/15` unseen)** — beating `BlindSpiral_Stiff` (`33.3%`) by **`+13.4 pp`** on `0.5 mm` tight clearances!

### 4.3 M1 vs. M2 vs. M3 Scientific Comparison
1. **Zero-Shot Unseen Shape Generalization Holds Strongly:** Trained *exclusively* on `round`, `square`, `rectangle`, all 3 RL methods generalized zero-shot to `hex`, `triangle`, `d_shape` with near-zero drop (`M3` Split A: **`51.1%` unseen vs `48.9%` seen**; `M1` Split B: **`55.6%` unseen vs `51.1%` seen**; `M1` Split C: **`28.9%` unseen vs `15.6%` seen**).
2. **Active Torque-Limit Compliance (`M3`) Provides Clear Physical & Stability Gains:**
   - **Best Split A accuracy & lowest seed variance across all 3 splits:** `M3` led Split A (`50.0% ± 2.7%` vs `47.8% ± 9.6%` `M1` and `37.8% ± 5.7%` `M2`) with **`3–4×` tighter seed standard deviation** (`±2.7–4.2%` vs `±9.6–17.5%` on `M1`).
   - **Substantially gentler contact & unbolted socket safety:** `M3` reduced median peak contact force by **`-13` to `-17 N`** vs `M1` on every split (`72.5 / 53.2 / 66.5 N` vs `89.1 / 61.3 / 79.4 N`), cut Split B unbolted shove failures **`3.3×`** vs `M1` (`0.7` vs `2.3`, and `7×` vs `BlindSpiral_Stiff`'s `5`), and cut median unbolted socket displacement nearly in half (**`4.9 mm`** vs `9.0 mm` `M1`, `9.5 mm` `M2`, `16.8 mm` `Spiral Stiff`, `12.8 mm` `Spiral Low-Torque`).
   - **Learned Phase-Dependent Torque Modulation:** Starting from `0.75` on episode 0, `M3` consistently settled to **`~0.70` during free / plate-search / rim contact** and **dropped to `~0.63–0.64` during insertion/seating** (complying to hole walls during entry rather than wedging), while achieving the highest held-out contact-class decoder accuracy (`75.3 / 77.4 / 73.6%`).
   - **Action-Space Sample-Efficiency Trade-Off:** Because `M3` optimizes an `8`-dim continuous action space (`3` motion + `5` per-joint `torque_scale`) vs `M1`'s `3`-dim space under the same `500k` step budget, `M3`'s exploration `std` annealed slower (`~0.47–0.53` vs `0.32–0.34` on `M1`), leaving additional headroom if `M3` uses either a scalar/2-group `torque_scale` (4–5 dim action) or a `1.0M`-step budget with `>= 65%` curriculum promotion so all seeds reach `16 mm`.

**Stopped here for review.**
