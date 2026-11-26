# Phase 3 Results: Gravity-Compensated Servo Torque Limiting (`GravComp-TL`) & Gravity-Subtracted Current Sensing (`τ_ext`, `Δτ_ext`)

**Project:** `projects/blind_insertion/` (Isolated Arm-Only `mj_rne` Gravity Compensation + `GravComp-TL` + `GRU(64)` Asymmetric Actor-Critic PPO at `2.0M` steps/run; [`geometry.py`](geometry.py), [`env.py`](env.py), [`benchmarks.py`](benchmarks.py), and [`controllers.py`](controllers.py) completely untouched)  
**Date:** 2026-11-24  
**Headline Verdict:** **ZERO-TRAINING `GravComp-TL` (`BlindSpiral α=0.35`) IS A MAJOR HARDWARE-CONTROL WIN — ELIMINATING THE `-30 pp` BOLTED GRAVITY-SAG STALL OF RAW `ts=0.35` (`70.0%` VS `43.3%` SPLIT A), CUTTING PEAK CONTACT FORCE BY `42–50%` ACROSS ALL 4 SPLITS (`34.9 / 25.2 / 28.7 / 15.0 N`), ACHIEVING `0` SHOVES ON SPLITS B & C, AND BOOSTING OOD SLIPPERY `+13.3 pp` (`40.0%` VS `26.7%`). IN THE `2.0M`-STEP RL SHOWDOWN (`0 / 0` SAFETY VIOLATIONS ACROSS `1,560` ROLLOUTS), `M4_GravComp_TL` CRUSHED FIXED-COMPLIANCE `M4_Fixed_GravComp` (`+23` TO `+30 pp`), CUT PEAK FORCES NEARLY IN HALF VS `M1_PosOnly_2M` (`27.3–40.5 N` VS `51.7–71.6 N`), ACHIEVED `0.0` SHOVES & `0.7–2.7 mm` SOCKET DISPLACEMENT ON SPLITS B & C, AND CUT `jammed_on_rim` BY `32–33%` VS PHASE 2 `M3` — YET COMPACT KINEMATIC-ONLY `M1_PosOnly_2M` STILL WON RAW SUCCESS RATE ACROSS ALL 4 SPLITS (`50.0 / 58.3 / 31.7 / 43.3%`, WITH `s101` REACHING `76.7%` SPLIT B & `53.3%` OOD SLIPPERY WITH `6` SHOVES) BECAUSE 1-STEP FINITE-DIFFERENCING NOISY `20 ms`-LAGGED STS3215 CURRENT (`Δτ_ext`) AMPLIFIED SENSOR NOISE (`aux` `65%` VS `81%`) AND REDUCED EPISODE TURNOVER.**

**Code & artifacts:**
- Arm-only `compute_arm_gravity_torque(q_quantized)` (`0.0066 ms/step`) + `GravCompBlindInsertionEnv` wrapper: [`phase3_gravcomp.py`](phase3_gravcomp.py)
- `GRU(64)` Phase 3 policy (`M1_PosOnly_2M`, `M4_GravComp_TL`, `M4_Fixed_GravComp`) + `tau_ext / d_tau_ext` feature builder: [`phase3_policy.py`](phase3_policy.py)
- Zero-training physics gate + `2.0M`-step RL trainer & 4-suite evaluator: [`phase3_train.py`](phase3_train.py)
- Unit tests (`21/21 PASS`, `5.57 s`): [`test_phase3.py`](test_phase3.py), [`test_phase2.py`](test_phase2.py), [`test_phase1.py`](test_phase1.py), [`test_phase0.py`](test_phase0.py)
- Saved checkpoints (`6` `.npz` files with frozen `RunningMeanStd`): `phase3_checkpoints/`
- Full machine-readable results: [`phase3_results.json`](phase3_results.json) (plus gate table [`phase3_gate.json`](phase3_gate.json))

---

## 1. Hard-Constraint, Physics & Safety Verification

| Check | Result |
| --- | --- |
| [`geometry.py`](geometry.py), [`env.py`](env.py), [`benchmarks.py`](benchmarks.py) (Splits A / B / C), [`controllers.py`](controllers.py) untouched | **PASS** (`git diff --name-only` empty on all four) |
| Isolated arm-only `compute_arm_gravity_torque(q_quantized)` (`mj_rne`, zero peg/socket/table) invariance & latency | **PASS** (`atol=1e-12` invariant across shapes/masses/contacts; **`0.00655 ms/step`** vs `< 0.1 ms` gate) |
| `GravComp-TL` hardware clamping `lim_i = clip(|tau_grav_i| + alpha_i * 1.50, 0.05, 2.94)` & zero gravity sag at `alpha=0.04` | **PASS** (unit-tested in MuJoCo: `|dz| < 3.5 mm` in free hover; exact force clamping verified) |
| `tau_ext` / `d_tau_ext` controller-legality (`d_tau_ext=0` at step 0), `M1` bit-invariance, & `M4` zero-init `alpha=0.35` | **PASS** (unit-tested) |
| Strict Phase 2 training distribution preserved (`[0.35, 0.55]` `mu_table`, `[180, 320] g`, seen shapes only, `78,160` train episodes, seeds `>= 250000`) | **PASS** (OOD Slippery `mu=0.25, 120 g` remains strictly OOD) |
| Unit tests (`test_phase0.py` – `test_phase3.py`) | **21/21 PASS** (`5.57 s`) |
| Evaluation safety violations (`self_collision_steps` / `joint_limit_steps` across all `1,560` held-out rollouts) | **`0 / 0`** |

---

## 2. Step 2 Zero-Training Physics Verification Gate (`GravComp-TL` BlindSpiral)

> Evaluated across all 4 deterministic 30-trial suites (`120` trials per controller, `0 / 0` safety violations) before any RL training.

| Controller | Split A (`bolted`, `n=30`) | Split B (`unbolted`, `n=30`) | Split C (`tight_0.5mm`, `n=30`) | OOD Slippery (`mu=0.25, 120g`, `n=30`) |
| --- | ---: | ---: | ---: | ---: |
| **Nominal Straight-Down** (`ts=1.0`) | `0/30 (0.0%)` · `26.0 N` · `rim=12` | `2/30 (6.7%)` · `24.0 N` · `4` shv · `1.3 mm` | `2/30 (6.7%)` · `26.9 N` · `1` shv | `0/30 (0.0%)` · `12.6 N` · `28` shv |
| **Blind Spiral · Stiff** (`ts=1.0`) | `22/30 (73.3%)` · `70.3 N` · `rim=3` | `24/30 (80.0%)` · `43.7 N` · `5` shv · `16.8 mm` | `10/30 (33.3%)` · `56.9 N` · `4` shv · `13.7 mm` | `8/30 (26.7%)` · `27.4 N` · `22` shv · `25.2 mm` |
| **Blind Spiral · Raw Low-Torque** (`ts=0.35`) | `13/30 (43.3%)` · `46.6 N` · `rim=7` | `25/30 (83.3%)` · `31.0 N` · `0` shv · `12.8 mm` | `7/30 (23.3%)` · `41.6 N` · `1` shv · `8.3 mm` | `14/30 (46.7%)` · `19.3 N` · `15` shv · `24.5 mm` |
| **Blind Spiral · `GravComp α=0.10`** | `9/30 (30.0%)` · **`17.0 N`** · `rim=12` | `3/30 (10.0%)` · **`11.7 N`** · `0` shv · **`0.1 mm`** | `0/30 (0.0%)` · **`13.7 N`** · `0` shv · **`0.0 mm`** | `2/30 (6.7%)` · **`8.8 N`** · `14` shv · `21.6 mm` |
| **Blind Spiral · `GravComp α=0.20`** | `14/30 (46.7%)` · `27.4 N` · `rim=12` | `14/30 (46.7%)` · `18.1 N` · `0` shv · `2.6 mm` | `8/30 (26.7%)` · `21.0 N` · `0` shv · `0.1 mm` | `8/30 (26.7%)` · `11.1 N` · `15` shv · `24.7 mm` |
| **Blind Spiral · `GravComp α=0.35`** | **`21/30 (70.0%)`** · **`34.9 N`** · `rim=5` | **`22/30 (73.3%)`** · **`25.2 N`** · **`0` shv** · `10.3 mm` | **`10/30 (33.3%)`** · **`28.7 N`** · **`0` shv** · `7.2 mm` | **`12/30 (40.0%)`** · **`15.0 N`** · **`15` shv** · `24.0 mm` |
| **Privileged Oracle** | `29/30 (96.7%)` · `0.0 N` | `30/30 (100.0%)` · `8.1 N` · `0` shv · `0.3 mm` | `29/30 (96.7%)` · `10.9 N` · `0` shv · `0.5 mm` | `30/30 (100.0%)` · `9.3 N` · `0` shv · `1.3 mm` |

**Gate Verdict — PASSED:** `BlindSpiral_GravComp_0.35` eliminates the `-30 pp` bolted gravity-sag stall of raw `ts = 0.35` (`70.0%` vs `43.3%` Split A; `33.3%` vs `23.3%` Split C), **halves peak contact force across every split (`-42%` to `-50%`)**, achieves **`0` shoves on Splits B & C**, and gains **`+13.3 pp` (`-7` shoves)** on OOD Slippery over `BlindSpiral_Stiff`.

---

## 3. Converged `2.0M`-Step RL Showdown Across All 4 Suites (2-Seed Mean ± Std, `[s101, s202]`)

> All 3 RL methods trained identically for `2.0M` policy steps/seed × `2` seeds (`101, 202`) = `12.0M` total steps (`8,060.0 s` = `2.24 hr` on 10 CPU workers), sharing `GRU(64)`, `ee_rel_start` + `4×4` spatial memory, asymmetric critic, dual aux decoder, smooth `6.0 → 16.5 mm` curriculum, and Phase 2's exact reward & training distribution.

### Split A — `bolted_socket` (`n = 30`, `1.0–1.5 mm` clearance)

| Method | Overall Success | Seen / Unseen | `<= 10` / `> 10 mm` | `jammed_on_rim` (of 60) | Hole Reach (`succ+rim`) | Med. Time | Med. Peak Force | `α_lat` / `α_arm` (F / S / I) | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Blind Spiral · `GravComp α=0.35`** | `70.0%` (`21/30`) | `73.3%` / `66.7%` | — | `5/30` | `86.7%` | `1.90 s` | `34.9 N` | `0.35 / 0.35` | `0 / 0` |
| **M1 · `M1_PosOnly_2M`** *(2 seeds)* | **`50.0% ± 6.7%`** `[17, 13]` | `53.3%` / **`46.7%`** | **`66.7%` / `38.9%`** | **`16/60`** (`8.0`) | **`76.7%`** | `0.72 s` | `71.6 N` | `ts=1.00` | `0 / 0` |
| **M4 · `M4_GravComp_TL`** *(2 seeds)* | `36.7% ± 3.3%` `[10, 12]` | `43.3%` / `30.0%` | `50.0%` / `27.8%` | `19/60` (`9.5`) | `68.3%` | `0.74 s` | **`40.5 N`** | `0.38/0.47/0.36` · `0.44/0.42/0.55` | `0 / 0` |
| **M4 · `M4_Fixed_GravComp` (`α=0.20`)** | `6.7% ± 6.7%` `[0, 4]` | `3.3%` / `10.0%` | `4.2%` / `8.3%` | `31/60` (`15.5`) | `58.3%` | `2.75 s` | `24.8 N` | `0.20 / 0.20` | `0 / 0` |

### Split B — `unbolted_socket` (`n = 30`, `1.0–1.5 mm`, `180–320 g`, `mu ∈ [0.35, 0.55]`)

| Method | Overall Success | Seen / Unseen | `<= 10` / `> 10 mm` | `jammed_on_rim` (of 60) | Med. Time | Med. Peak Force | Shoves / Med. Disp | `α_lat` / `α_arm` (F / S / I) | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Blind Spiral · `GravComp α=0.35`** | `73.3%` (`22/30`) | `66.7%` / `80.0%` | — | `2/30` | `1.60 s` | `25.2 N` | `0` / `10.3 mm` | `0.35 / 0.35` | `0 / 0` |
| **M1 · `M1_PosOnly_2M`** *(2 seeds)* | **`58.3% ± 18.3%`** `[23, 12]` | **`53.3%` / `63.3%`** | **`58.3%` / `58.3%`** | **`11/60`** (`5.5`) | **`0.50 s`** | `51.7 N` | `0.5` / `6.4 mm` | `ts=1.00` | `0 / 0` |
| *↳ `M1_PosOnly_2M s=101`* | ***`76.7%` (`23/30`)*** | *`66.7%` / `86.7%`* | — | *`4/30`* | — | *`47.7 N`* | *`0` shv* | *`ts=1.00`* | *`0 / 0`* |
| **M4 · `M4_GravComp_TL`** *(2 seeds)* | `45.0% ± 11.7%` `[10, 17]` | `50.0%` / `40.0%` | `41.7%` / `47.2%` | `16/60` (`8.0`) | `0.69 s` | **`27.3 N`** | **`0.0` / `2.7 mm`** | `0.40/0.49/0.37` · `0.44/0.43/0.55` | `0 / 0` |
| **M4 · `M4_Fixed_GravComp` (`α=0.20`)** | `16.7% ± 0.0%` `[5, 5]` | `20.0%` / `13.3%` | `33.3%` / `5.6%` | `23/60` (`11.5`) | `1.30 s` | `18.9 N` | `0.0` / `0.1 mm` | `0.20 / 0.20` | `0 / 0` |

### Split C — `tight_0.5mm` (`n = 30`, `0.5 mm` clearance, `15` bolted + `15` unbolted)

| Method | Overall Success | Seen / Unseen | `<= 10` / `> 10 mm` | `jammed_on_rim` (of 60) | Med. Time | Med. Peak Force | Shoves / Med. Disp | `α_lat` / `α_arm` (F / S / I) | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Blind Spiral · Stiff** (`ts=1.0`) | `33.3%` (`10/30`) | `40.0%` / `26.7%` | `41.7%` / `27.8%` | `5/30` | `1.55 s` | `56.9 N` | `4` / `13.7 mm` | `ts=1.00` | `0 / 0` |
| **Blind Spiral · `GravComp α=0.35`** | `33.3%` (`10/30`) | `33.3%` / `33.3%` | — | `4/30` | `2.39 s` | `28.7 N` | `0` / `7.2 mm` | `0.35 / 0.35` | `0 / 0` |
| **M1 · `M1_PosOnly_2M`** *(2 seeds)* | **`31.7% ± 1.7%`** `[10, 9]` | `23.3%` / **`40.0%`** | `41.7%` / **`25.0%`** | `26/60` (`13.0`) | **`0.95 s`** | `62.1 N` | `0.5` / `4.9 mm` | `ts=1.00` | `0 / 0` |
| **M4 · `M4_GravComp_TL`** *(2 seeds)* | `28.3% ± 8.3%` `[6, 11]` | **`33.3%`** / `23.3%` | **`45.8%`** / `16.7%` | **`21/60`** (`10.5`) | `1.21 s` | **`34.2 N`** | **`0.0` / `0.7 mm`** | `0.37/0.46/0.39` · `0.43/0.44/0.54` | `0 / 0` |
| *↳ `M4_GravComp_TL s=202`* | ***`36.7%` (`11/30`)*** | *`53.3%` / `20.0%`* | — | *`10/30`* | — | *`37.4 N`* | *`0` shv* | — | *`0 / 0`* |
| **M4 · `M4_Fixed_GravComp` (`α=0.20`)** | `5.0% ± 1.7%` `[1, 2]` | `3.3%` / `6.7%` | `12.5%` / `0.0%` | `35/60` (`17.5`) | `2.84 s` | `22.4 N` | `0.0` / `0.1 mm` | `0.20 / 0.20` | `0 / 0` |

### OOD Slippery Unbolted Ablation (`mu_table = 0.25`, `120 g`, breakaway `0.29 N`, `n = 30`)

| Method | Overall Success | Seen / Unseen | `jammed_on_rim` (of 60) | Med. Time | Med. Peak Force | Shove Failures | Med. Socket Disp | `α_lat` / `α_arm` (F / S / I) | Safety |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Blind Spiral · Stiff** (`ts=1.0`) | `26.7%` (`8/30`) | `26.7%` / `26.7%` | `0/30` | `1.28 s` | `27.4 N` | `22 / 30` | `25.2 mm` | `ts=1.00` | `0 / 0` |
| **Blind Spiral · Low-Torque** (`ts=0.35`) | `46.7%` (`14/30`) | `40.0%` / `53.3%` | `0/30` | `1.67 s` | `19.3 N` | `15 / 30` | `24.5 mm` | `ts=0.35` | `0 / 0` |
| **Blind Spiral · `GravComp α=0.35`** | `40.0%` (`12/30`) | `26.7%` / `53.3%` | `1/30` | `1.96 s` | `15.0 N` | `15 / 30` | `24.0 mm` | `0.35 / 0.35` | `0 / 0` |
| **M1 · `M1_PosOnly_2M`** *(2 seeds)* | **`43.3% ± 10.0%`** `[16, 10]` | **`36.7%` / `50.0%`** | `5/60` | `0.76 s` | `32.9 N` | `11.5 / 30` `[6, 17]` | `20.2 mm` | `ts=1.00` | `0 / 0` |
| *↳ `M1_PosOnly_2M s=101`* | ***`53.3%` (`16/30`)*** | *`53.3%` / `53.3%`* | *`4/30`* | — | *`33.4 N`* | ***`6 / 30`*** | — | *`ts=1.00`* | *`0 / 0`* |
| **M4 · `M4_GravComp_TL`** *(2 seeds)* | `15.0% ± 1.7%` `[4, 5]` | `16.7%` / `13.3%` | `6/60` | `0.67 s` | **`19.0 N`** | `15.5 / 30` `[14, 17]` | `20.3 mm` | `0.39/0.48/0.39` · `0.44/0.43/0.53` | `0 / 0` |
| **M4 · `M4_Fixed_GravComp` (`α=0.20`)** | `6.7% ± 0.0%` `[2, 2]` | `0.0%` / `13.3%` | `19/60` | `1.18 s` | **`13.8 N`** | **`8.0 / 30`** `[14, 2]` | **`12.6 mm`** | `0.20 / 0.20` | `0 / 0` |

---

## 4. Compute, Training Stability & Learned `α` Modulation

| RL Method | Obs / Act Dim | Mean Steps / Seed | Mean Episodes / Seed | Mean Wall / Seed | Throughput | Train Off-Plate (Last 10 Upd) | Fail Med. `max_depth` (A / B / C) | Held-Out Aux Cls Acc (A / B / C / Slip) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **M1 · `M1_PosOnly_2M`** | `69 / 3` | `2,002,446` | `16,110` | `1,328.7 s` | `1,513 st/s` | **`[0.0, 0.0]%`** | `0.2 / 0.2 / 0.2 mm` | **`80.7 / 77.9 / 78.8 / 70.9%`** |
| **M4 · `M4_GravComp_TL`** | `99 / 5` | `2,001,578` | `12,670` | `1,225.6 s` | **`1,638 st/s`** | **`[0.0, 0.0]%`** | `0.1 / 0.1 / 0.1 mm` | `65.3 / 72.8 / 66.7 / 58.4%` |
| **M4 · `M4_Fixed_GravComp`** | `97 / 3` | `2,001,668` | `10,300` | `1,370.0 s` | `1,461 st/s` | **`[0.0, 0.0]%`** | `0.1 / 0.1 / 0.1 mm` | `69.2 / 75.0 / 72.1 / 64.0%` |

---

## 5. Ground-Truth Scientific Analysis (What Worked, What Failed, and Why)

1. **Hardware Physics — `GravComp-TL` Works Decisively:**
   - Subtracting nominal arm gravity `|tau_grav(q_quant)|` (`0.0066 ms/step`, zero privileged info) before adding contact margin `alpha * 1.50 N·m` completely fixes the SO-101 gravity-sag bottleneck: `BlindSpiral_GravComp_0.35` recovers `+26.7 pp` over raw `ts=0.35` on bolted Split A (`70.0%` vs `43.3%`), cuts peak contact forces by `42–50%` across every split, and eliminates all unbolted shoves on Splits B & C (`0` vs `5` and `0` vs `4`).
2. **Dynamic `α` Modulation (`M4_GravComp_TL`) vs. Fixed Low `α = 0.20` (`M4_Fixed_GravComp`):**
   - Locking `alpha = 0.20` (`0.30 N·m` margin) starved RL of friction-breaking authority: `M4_Fixed_GravComp` stalled in `jammed_on_rim` on `31/60` (Split A) and `35/60` (Split C), scoring only `5.0–16.7%`.
   - Giving the policy dynamic 2-group `[alpha_lat, alpha_arm]` control (`M4_GravComp_TL`) added **`+30.0 pp` Split A / `+28.3 pp` Split B / `+23.3 pp` Split C**, learned clear phase-dependent compliance (**higher `alpha_lat = 0.46–0.49` during plate drag, dropping to `0.36–0.39` during hole entry; lower `alpha_arm = 0.42–0.44` during search, ramping to `0.54–0.55` during seating**), cut peak forces by **`42–47%` vs `M1_PosOnly_2M`** (`40.5 / 27.3 / 34.2 / 19.0 N`), reduced Split B/C socket displacement to **`2.7 mm` / `0.7 mm`** with **`0.0` shoves across all `120` Split B & C trials**, and cut `jammed_on_rim` by **`32–33%` vs Phase 2 `M3`**.
3. **Why `M1_PosOnly_2M` Still Beat `M4_GravComp_TL` on Raw Success Rate (Including OOD Slippery):**
   - **Noise amplification from 1-step `Δτ_ext` on `20 ms`-lagged noisy current:** Because `actuator_force_noisy` carries `±8%` multiplicative + `0.02 N·m` additive per-step white noise, taking the 1-step finite difference `d_tau_ext = tau_ext(t) - tau_ext(t-1)` amplifies sensor noise by `√2` while still lagging real contact by `20 ms`. That dropped `M4`'s auxiliary contact-state accuracy to `65.3 / 72.8 / 66.7 / 58.4%` vs `M1_PosOnly_2M`'s **`80.7 / 77.9 / 78.8 / 70.9%`**!
   - **Sample efficiency & episode turnover (`16,110` vs `12,670` episodes in `2.0M` steps):** With a compact 69-dim clean kinematic input and 3-dim action space at full `2.94 N·m` stiffness, `M1_PosOnly_2M` moved briskly across dry plate friction, completed `27%` more episodes in `2.0M` steps, cut its own `jammed_on_rim` failures dramatically (`25 → 16` Split A, `18 → 11` Split B), and reached **`50.0%` A / `58.3%` B (`s101 = 76.7%, 0 shoves`) / `43.3%` OOD Slippery (`s101 = 53.3%, 6 shoves`)** purely via learned gentle-press kinematic motion.
   - **OOD Slippery Training Distribution Ceiling:** Because training kept `mu_table ∈ [0.35, 0.55]` (`180–320 g`, where `alpha_lat ≈ 0.48` already gave `0.0` shoves on Splits B & C), `M4_GravComp_TL` converged to `alpha_lat ≈ 0.48` (`0.72 N·m` → `~3.3 N` lateral force) and still shoved `15.5/30` sockets on `0.29 N` OOD Slippery (`15.0%`), whereas `M4_Fixed_GravComp (α=0.20)` halved OOD shoves to `8.0/30` (`13.8 N`, `12.6 mm` disp).
4. **Bottom-Line Architectural Takeaway:** On low-cost STS3215 arms with `20 ms` current lag and `±8%` noise, **`GravComp-TL` belongs in the low-level actuator compliance layer** (cutting contact force `42–50%` and shoves to `0.0` with zero gravity sag), while **clean kinematic-only encoders (`M1`) remain the highest-SNR policy input** unless motor-current features are low-pass filtered / denoised rather than 1-step differenced (`Δτ_ext`) and paired with a wider friction/mass training range.

**Stopped here for review.**
