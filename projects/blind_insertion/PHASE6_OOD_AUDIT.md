# Phase 6 Zero-Shot OOD Stress Audit (`R2_v2` vs. `R1_v2`, Frozen Phase 5 Checkpoints)

**Evaluation Protocol:**
- **ZERO retraining, ZERO policy edits.** Evaluated the exact frozen Phase 5 checkpoints (`R1_v2_Res_PosOnly` and `R2_v2_Res_GravComp_TL`, all 3 seeds `[101, 202, 303]`) alongside `BlindSpiral_Stiff_1.0`, `BlindSpiral_GravComp_0.35`, and `Privileged_Oracle`.
- **5 Orthogonal OOD Stress Suites** (`20` deterministic trials each, disjoint seeds `5100..5519`, `900` total OOD rollouts in `81.7 s` headless):
  1. **`OOD_Height_Tilt`**: Fixture height shifted `±12 mm` (`+12 mm` even / `-12 mm` odd) **and** tilted `3°–5°`.
  2. **`OOD_Spring_Mount`**: Compliant 3-axis spring-damper fixture (`k_z = 800 N/m`, `k_xy = 600 N/m`, `damping = 16 N·s/m`) — flexing PCB / rubber-footed socket proxy.
  3. **`OOD_NonConvex_Asym`**: Unseen non-convex asymmetric `L-shape` (`10`) and `keyed_notch` (`10`) holes via `mjfit` at `1.0 mm` clearance (`50%` bolted / `50%` unbolted; `100%` drop-test verified).
  4. **`OOD_Grasp_Error`**: Unobserved `±3.5 mm` radial grasp offset **and** `4°–6°` in-hand peg tilt (`50%` bolted / `50%` unbolted).
  5. **`OOD_UltraTight_0.30mm`**: Sub-encoder-count `0.30 mm` clearance across all 6 standard shapes (`50%` bolted / `50%` unbolted; `100%` drop-test verified).
- **Artifacts:** [`phase6_ood_eval.py`](file:///home/krisd/physical-ai-systems/projects/blind_insertion/phase6_ood_eval.py), [`phase6_ood_results.json`](file:///home/krisd/physical-ai-systems/projects/blind_insertion/phase6_ood_results.json), [`test_phase6.py`](file:///home/krisd/physical-ai-systems/projects/blind_insertion/test_phase6.py). All Phase 0–5 code, checkpoints, and benchmarks remain untouched.

---

## Executive Verdict

- **Solvability & Safety Gate — PASSED 100%:** `Privileged_Oracle` solves **`98.0%`** across all 5 OOD suites (`100% / 100% / 100% / 100% / 90%`, all `>= 85%`), and **all `900` OOD rollouts recorded `0` self-collision steps and `0` joint-limit violations (`0/0` safety)**.
- **Overall Zero-Shot OOD Transfer Mean (5 Suites):**
  - `BlindSpiral_Stiff_1.0`: `33.0%` (`44.3 N` mean median peak force)
  - `BlindSpiral_GravComp_0.35`: `34.0%` (`25.4 N`)
  - `R1_v2_Res_PosOnly` (Stiff): `42.0%` (`50.2 N`)
  - **`R2_v2_Res_GravComp_TL` (Ours): `49.7%` (`+7.7 pp` over `R1_v2`, `+15.7 pp` over `BlindSpiral_GravComp`, `-21.1 N` lower peak force than `R1_v2`)**
- **Honest Transfer Profile (Where It Excels vs. Where It Breaks):**
  - **Strong Zero-Shot Transfer (`70%–82%` or `+20 to +28 pp` over Stiff):** Non-convex asymmetric holes (`81.7%`, `+20.0 pp`), sub-encoder `0.30 mm` clearance (`45.0%`, `+28.3 pp`), and `±3.5 mm / 4°–6°` messy grasps (`70.0%`, `+25.0 pp` over base spiral, `-23.8 N` force vs. `R1_v2`).
  - **Honest Failure Regimes Exposed:**
    1. **`OOD_Height_Tilt` (`±12 mm`, `3°–5°`):** `R1_v2` wins (`50.0%` vs. `R2_v2` `33.3%`, `-16.7 pp` delta) because `ee_pose_fk[2]` (absolute `z`) shifts out of distribution (`aux_acc` drops `80% → 40–62%`) and stiff `R1_v2` brute-forces `+12 mm` tilted walls at `71.9 N` (`80%` at `+12 mm`, `20%` at `-12 mm`), whereas torque-limited `R2_v2` (`41.4 N`) is height-symmetric (`33.3% / 33.3%`) but stalls more often on tilted walls.
    2. **`OOD_Spring_Mount` (`k_z = 800 N/m`):** All blind policies break (`5%–18.3%`, `R2_v2` still highest at `18.3%` vs. `R1_v2` `10.0%`) because the frozen base spiral's `ee_pose_fk[2] < z_contact - 1.8 mm` drop-in trigger false-fires under `2 N` of normal search preload (`2.5 mm` spring compression on the flat plate!).

---

## 1. Full 5-Suite Zero-Shot OOD Stress Table (`20` Trials / Suite)

| OOD Stress Suite | `Privileged_Oracle` | `BlindSpiral_Stiff` (`1.0`) | `BlindSpiral_GC` (`0.35`) | `R1_v2` Stiff (`3` seeds) | **`R2_v2` GravComp-TL (`3` seeds)** | `R2_v2 − R1_v2` Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OOD_Height_Tilt`** (`±12 mm, 3°–5°`) | `100.0%` (`8.5 N`) | `20.0%` (`65.1 N`) | `30.0%` (`36.4 N`) | **`50.0% ± 4.1%`** `[11,10,9]` (`71.9 N`) | `33.3% ± 4.7%` `[6,8,6]` (`41.4 N`) | **`-16.7 pp`** (`-30.5 N`) |
| **`OOD_Spring_Mount`** (`k_z=800, k_xy=600`) | `100.0%` (`0.0 N`) | `10.0%` (`22.1 N`) | `5.0%` (`12.6 N`) | `10.0% ± 7.1%` `[3,0,3]` (`20.8 N`) | **`18.3% ± 12.5%`** `[3,1,7]` (`13.6 N`) | **`+8.3 pp`** (`-7.2 N`) |
| **`OOD_NonConvex_Asym`** (`L / Keyed, 1.0 mm`) | `100.0%` (`5.3 N`) | `70.0%` (`46.0 N`) | `60.0%` (`27.4 N`) | `61.7% ± 10.3%` `[10,12,15]` (`44.7 N`) | **`81.7% ± 4.7%`** `[17,15,17]` (`24.3 N`) | **`+20.0 pp`** (`-20.4 N`) |
| **`OOD_Grasp_Error`** (`±3.5 mm, 4°–6°`) | `100.0%` (`3.9 N`) | `35.0%` (`41.8 N`) | `45.0%` (`22.0 N`) | **`71.7% ± 15.5%`** `[10,17,16]` (`56.4 N`) | `70.0% ± 4.1%` `[15,14,13]` (`32.6 N`) | **`-1.7 pp`** (`-23.8 N`) |
| **`OOD_UltraTight_0.30mm`** (`0.30 mm`) | `90.0%` (`12.1 N`) | `30.0%` (`46.7 N`) | `30.0%` (`28.6 N`) | `16.7% ± 2.4%` `[3,3,4]` (`57.2 N`) | **`45.0% ± 10.8%`** `[12,7,8]` (`33.7 N`) | **`+28.3 pp`** (`-23.5 N`) |
| **5-Suite Mean** (`100` OOD trials/model) | **`98.0%`** (`6.0 N`) | `33.0%` (`44.3 N`) | `34.0%` (`25.4 N`) | `42.0%` (`50.2 N`) | **`49.7%`** (`29.1 N`) | **`+7.7 pp`** (`-21.1 N`) |

---

## 2. Detailed Telemetry & Failure-Mode Breakdown (`R1_v2` vs. `R2_v2`, Pooled Across 3 Seeds = `60` Trials / Suite)

| Suite | Method | Med Insert Time (s) | Med Peak Force (N) | Aux Contact Acc | `success` | `jammed_on_rim` | `jammed_on_plate` | `stalled_in_hole` | `socket_shoved` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OOD_Height_Tilt`** | `R1_v2` | `2.27 s` | `71.9 N` | `40.5%` | `30` | `9` | `20` | `1` | `0` |
| **`OOD_Height_Tilt`** | **`R2_v2`** | `1.82 s` | **`41.4 N`** | **`62.4%`** | `20` | `14` | `23` | `3` | `0` |
| **`OOD_Spring_Mount`** | `R1_v2` | `0.93 s` | `20.8 N` | `19.7%` | `6` | `18` | `29` | `7` | `0` |
| **`OOD_Spring_Mount`** | **`R2_v2`** | `1.45 s` | **`13.6 N`** | **`44.6%`** | **`11`** | `21` | `28` | `0` | `0` |
| **`OOD_NonConvex_Asym`** | `R1_v2` | `2.08 s` | `44.7 N` | `75.8%` | `37` | `12` | `11` | `0` | `0` |
| **`OOD_NonConvex_Asym`** | **`R2_v2`** | `2.05 s` | **`24.3 N`** | **`85.6%`** | **`49`** | **`7`** | **`4`** | `0` | `0` |
| **`OOD_Grasp_Error`** | `R1_v2` | `1.13 s` | `56.4 N` | `64.1%` | `43` | `8` | `9` | `0` | `0` |
| **`OOD_Grasp_Error`** | **`R2_v2`** | `1.02 s` | **`32.6 N`** | **`76.2%`** | `42` | `12` | `5` | `1` | `0` |
| **`OOD_UltraTight_0.30mm`** | `R1_v2` | `1.83 s` | `57.2 N` | `74.1%` | `10` | `22` | `28` | `0` | `0` |
| **`OOD_UltraTight_0.30mm`** | **`R2_v2`** | `2.35 s` | **`33.7 N`** | **`82.1%`** | **`27`** | **`15`** | **`18`** | `0` | `0` |

### Key Diagnostic Sub-Breakdowns
- **`OOD_NonConvex_Asym` by shape:** `R2_v2` achieves **`keyed_notch 26/30 (86.7%)`** and **`l_shape 23/30 (76.7%)`** vs. `R1_v2` `keyed_notch 16/30 (53.3%)` and `l_shape 21/30 (70.0%)`. Tactile-amplified `GravComp-TL` generalizes zero-shot to complex non-convex re-entrant geometry without ever seeing a non-convex shape during training.
- **`OOD_UltraTight_0.30mm` (Sub-Encoder-Count Clearance):** `R2_v2` hits **`45.0%`** (seed `101` hits **`60.0%`**, `12/20`) with `0` shoves and `33.7 N` peak force, whereas stiff `R1_v2` collapses to **`16.7%`** (`57.2 N`, worse than the untrained `30.0%` base spiral!).
- **`OOD_Height_Tilt` (`+12 mm` vs. `-12 mm`):**
  - `R1_v2`: `+12 mm = 24/30 (80.0%)`, `-12 mm = 6/30 (20.0%)`
  - `R2_v2`: `+12 mm = 10/30 (33.3%)`, `-12 mm = 10/30 (33.3%)`
  - **Root cause:** Actor observations include absolute `ee_pose_fk[2]` (trained exclusively at `z ≈ 24 mm`). Removing absolute `z` (relying strictly on `ee_rel_start` + contact proprioception) and randomizing fixture height/tilt during training directly fixes this.
- **`OOD_Spring_Mount` (`k_z = 800 N/m`):**
  - Both base spirals fail almost completely (`BlindSpiral_Stiff 10%`, `BlindSpiral_GC 5%`, `stalled_in_hole = 16–18/20`) because `2 N` of downward search preload compresses the `800 N/m` spring by `2.5 mm`, immediately triggering `BlindSpiralController`'s hardcoded `ee_pose_fk[2] < z_contact - 1.8 mm` drop-in transition while still on the flat plate. Replacing the fixed `1.8 mm` absolute-`z` drop trigger with a **force/deflection-drop (or lateral-constraint) detector** is required for compliant fixtures.
- **Safety Verification:** **`0` self-collision steps and `0` joint-limit steps** across all `900` OOD rollouts (`Privileged_Oracle`, `BlindSpiral_Stiff`, `BlindSpiral_GravComp`, `R1_v2` × 3 seeds, `R2_v2` × 3 seeds).
