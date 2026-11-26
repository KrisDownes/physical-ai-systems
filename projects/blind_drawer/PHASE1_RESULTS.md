# Phase 1 Results — Blind Drawer Opening (`projects/blind_drawer/`)

**Experiment:** Cross-task generalization test of **Gravity-Compensated Servo Torque Limiting (`GravComp-TL`) + Relative-Kinematics `GRU(64)` Residual PPO** (`D2_Res_GravComp_TL`) vs. matched **Stiff Position Residual PPO** (`D1_Res_Stiff`) on blind unbolted sticky/latched drawer opening (`robotstudio_so101`, MuJoCo `1 kHz` physics / `50 Hz` policy, 12-bit STS3215 quantization + `0.0045 rad` backlash).

**Untouched Files Verified:** [env.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/env.py), [gravcomp.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/gravcomp.py), [benchmarks.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/benchmarks.py), [controllers.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/controllers.py), and all of `projects/blind_insertion/` were kept 100% untouched.

---

## 1. Executive Summary

- **Split A (`bolted_sticky`, unknown rail yaw `±18°` + `5–9 N` sticky detent) — `D2` wins decisively (+26.7 pp):** `D2_Res_GravComp_TL` achieves **`60.0% ± 2.7%`** (`[17, 18, 19]/30`) vs. `D1_Res_Stiff` **`33.3% ± 7.2%`** (`[9, 8, 13]/30`), cutting `off_axis_bind_stall` by **60%** (`60 → 24` across 90 trials) and solving **`93.8%`** (`45/48`) of `|yaw| <= 10°` and **`21.4%`** (`9/42`) of `|yaw| > 10°` (where `D1` gets **`0/42 = 0.0%`**).
- **Split B (`unbolted_cabinet`, `400–750 g`, `μ ∈ [0.40, 0.65]`) — Parity in mean success, 12× tighter variance & lower cabinet disturbance for `D2`:** `D2` achieves **`75.6% ± 1.6%`** (`[23, 23, 22]/30`, `7.2 mm` median cabinet displacement, `0.28°` tilt) vs. `D1` **`76.7% ± 18.9%`** (`[15, 27, 27]/30`, `9.8 mm` displacement, `0.45°` tilt; seed `101` collapsed to `50.0%`).
- **Split OOD (`ood_light_height_angle`: `280–350 g, μ=0.35, ±15 mm height, ±22° yaw`, 100% zero-shot) — Honest negative result for `D2` (`20.0%` vs. `D1` `35.0%`):** Both policies are **height-invariant** (`+15 mm` vs. `-15 mm` balanced, fixing the absolute-`z` vulnerability seen in `blind_insertion` Phase 6) and `D2` retains **+8.8 pp higher OOD contact-class accuracy** (`61.9%` vs. `53.1%`), **but `D2` suffers `47/60` (`15.7/20`) `cabinet_shoved` failures vs. `D1` `22/60` (`7.3/20`)**. Root cause: on `280–350 g, μ=0.35` cabinets (`μ·m·g ≈ 1.03 N` vs. `3–7 N` pull), anchoring the cabinet requires strong downward normal force (`-F_z`). Because `D2` couples vertical pinning and axial pull under one `α_arm` cap that stayed near `0.36`, it clips downward pinning force (`0.37°` tilt vs. `D1` `0.77°` and `Oracle` `1.79°`).
- **Safety & Compute:** **`0/0`** self-collisions and joint-limit violations across all `9,000,000+` training steps and `480` RL eval + `320` baseline eval rollouts (`10/10` unit tests pass). Peak total RSS stayed flat at **`0.78–0.82 GB`** across parent + 8 workers (`0 B` swap), total wall time **`9,695.9 s`** (`2.69 h`, `933–944 st/s`).

---

## 2. Benchmark Results (3 Seeds `[101, 202, 303]`, `1.5M` Steps/Run)

### 2.1 Success Rate & Physical Metrics Across All 3 Splits

| Method | Split A: `bolted_sticky` (n=30) | Split B: `unbolted_cabinet` (n=30) | Split OOD: `light_height_angle` (n=20) | Safety (`self_col / jlim`) |
| :--- | :---: | :---: | :---: | :---: |
| `Nominal_Stiff_1.0` *(Phase 0)* | `10.0%` (`3/30`) · `25.2 N` | `70.0%` (`21/30`) · `12.3 mm` · `0.67°` | `30.0%` (`6/20`) · `21.3 mm` · `0.95°` | `0 / 0` |
| `Nominal_RawLowTorque_0.35` *(Phase 0)* | `10.0%` (`3/30`) · `22.4 N` | `36.7%` (`11/30`) · `25.1 mm` · `0.11°` | `5.0%` (`1/20`) · `25.3 mm` · `0.12°` | `0 / 0` |
| `Nominal_GravComp_0.35` *(Phase 0)* | `13.3%` (`4/30`) · `23.3 N` | `56.7%` (`17/30`) · `17.4 mm` · `0.11°` | `10.0%` (`2/20`) · `25.3 mm` · `0.12°` | `0 / 0` |
| **`D1_Res_Stiff`** *(3-seed mean ± std)* | **`33.3% ± 7.2%`** `[9, 8, 13]` | **`76.7% ± 18.9%`** `[15, 27, 27]` | **`35.0% ± 12.2%`** `[7, 10, 4]` | **`0 / 0`** |
| **`D2_Res_GravComp_TL`** *(3-seed mean ± std)* | **`60.0% ± 2.7%`** `[17, 18, 19]` | **`75.6% ± 1.6%`** `[23, 23, 22]` | **`20.0% ± 4.1%`** `[5, 4, 3]` | **`0 / 0`** |
| **Delta (`D2 − D1`)** | **`+26.7 pp`** *(1.8×)* | **`−1.1 pp`** *(12× lower std)* | **`−15.0 pp`** *(see §4)* | `0 / 0` |
| `Privileged_Oracle` *(Phase 0 upper bound)* | `100.0%` (`30/30`) · `49.5 N` | `100.0%` (`30/30`) · `0.5 mm` · `1.19°` | `100.0%` (`20/20`) · `4.6 mm` · `1.79°` | `0 / 0` |

### 2.2 Detailed Per-Split Physical & Auxiliary Decoder Metrics (3-Seed Mean)

| Method | Split | Med. Travel (mm) | Med. Time (s) | Med. Peak Force (N) | Med. Cab. Disp (mm) | Med. Cab. Tilt (°) | Aux Contact Acc (%) | Aux Rail Yaw MAE (°) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `D1_Res_Stiff` | Split A | `50.4` | `2.55` | `35.7` | `0.0` | `0.00` | `97.9%` | `0.65°` |
| `D2_Res_GravComp_TL` | Split A | **`70.0`** | `2.53` | `36.1` | `0.0` | `0.00` | `83.2%` | `1.13°` |
| `D1_Res_Stiff` | Split B | `69.9` | `2.61` | `29.4` | `9.8` | `0.45` | `86.3%` | `0.64°` |
| `D2_Res_GravComp_TL` | Split B | **`70.4`** | **`2.43`** | `33.9` | **`7.2`** | **`0.28`** | `76.9%` | `1.33°` |
| `D1_Res_Stiff` | Split OOD | `67.5` | `2.89` | `27.8` | `21.2` | `0.77` | `53.1%` | `2.02°` |
| `D2_Res_GravComp_TL` | Split OOD | `31.9` | `2.65` | `29.6` | `25.4` | `0.37` | **`61.9%`** | `2.61°` |

### 2.3 Failure Mode Breakdown (Totals Across All 3 Seeds: `N=90` for Splits A/B, `N=60` for Split OOD)

| Method | Split | `success` | `off_axis_bind_stall` | `detent_stall` | `cabinet_shoved` | `cabinet_tipped` | `handle_slip` | `timeout_partial` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `D1_Res_Stiff` | Split A (`N=90`) | `30` | `60` | `0` | `0` | `0` | `0` | `0` |
| `D2_Res_GravComp_TL` | Split A (`N=90`) | **`54`** | **`24`** | `12` | `0` | `0` | `0` | `0` |
| `D1_Res_Stiff` | Split B (`N=90`) | `69` | `12` | `1` | `3` | `0` | `0` | `5` |
| `D2_Res_GravComp_TL` | Split B (`N=90`) | `68` | **`0`** | **`0`** | `21` | `0` | `0` | `1` |
| `D1_Res_Stiff` | Split OOD (`N=60`) | `21` | `6` | `0` | `22` | `0` | `0` | `11` |
| `D2_Res_GravComp_TL` | Split OOD (`N=60`) | `12` | **`0`** | `0` | `47` | `0` | `0` | `1` |

*(Zero `cabinet_tipped` and zero `handle_slip` failures occurred in any run.)*

---

## 3. Per-Seed Results & Learned `[α_lat, α_arm]` Torque Scheduling

### 3.1 Per-Seed Evaluation Table

| Method | Seed | Steps | Wall (s) | Throughput | Peak RSS | Split A | Split B | Split OOD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `D1_Res_Stiff` | `101` | `1,500,483` | `1,500.9` | `999.7 st/s` | `0.781 GB` | `30.0%` (`9/30`) | `50.0%` (`15/30`) | `35.0%` (`7/20`) |
| `D1_Res_Stiff` | `202` | `1,501,518` | `1,564.0` | `960.1 st/s` | `0.784 GB` | `26.7%` (`8/30`) | `90.0%` (`27/30`) | `50.0%` (`10/20`) |
| `D1_Res_Stiff` | `303` | `1,500,804` | `1,720.0` | `872.6 st/s` | `0.781 GB` | `43.3%` (`13/30`) | `90.0%` (`27/30`) | `20.0%` (`4/20`) |
| `D2_Res_GravComp_TL` | `101` | `1,501,774` | `1,694.0` | `886.5 st/s` | `0.818 GB` | `56.7%` (`17/30`) | `76.7%` (`23/30`) | `25.0%` (`5/20`) |
| `D2_Res_GravComp_TL` | `202` | `1,502,769` | `1,566.4` | `959.4 st/s` | `0.815 GB` | `60.0%` (`18/30`) | `76.7%` (`23/30`) | `20.0%` (`4/20`) |
| `D2_Res_GravComp_TL` | `303` | `1,501,281` | `1,577.1` | `951.9 st/s` | `0.814 GB` | `63.3%` (`19/30`) | `73.3%` (`22/30`) | `15.0%` (`3/20`) |

### 3.2 Learned Dynamic `[α_lat, α_arm]` by Contact Phase (`D2_Res_GravComp_TL`, 3-Seed Mean)

| Split | `free_slide` `(α_lat, α_arm)` | `off_axis_bind` `(α_lat, α_arm)` | `detent_catch` `(α_lat, α_arm)` | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Split A (`bolted_sticky`)** | `(0.350, 0.381)` | **`(0.301, 0.437)`** | **`(0.323, 0.434)`** | Drops `α_lat` during bind (`0.350 → 0.301`) to relieve lateral rail jam while stiffening `α_arm` (`0.381 → 0.437`, up to `0.480` on `s=303`) to clear detents. |
| **Split B (`unbolted_cabinet`)** | `(0.346, 0.360)` | `(0.365, 0.346)` | `(0.334, 0.373)` | Stays conservative near init `(0.35, 0.35)` — enough for `400–750 g` cabinets (`75.6%`, `7.2 mm` disp), zero bind stalls. |
| **Split OOD (`light_height_angle`)** | `(0.358, 0.354)` | `(0.365, 0.388)` | `(0.340, 0.359)` | Does **not** ramp `α_arm` high enough (`~0.36`) to supply downward pinning force on `280–350 g, μ=0.35` cabinets. |

---

## 4. Mechanism Audit: Why `D2` Wins Split A & Loses Split OOD

### 4.1 Split A Off-Axis Yaw Breakdown (`|yaw| <= 10°` vs. `|yaw| > 10°`)

| Method | `|yaw| <= 10°` | `|yaw| > 10°` |
| :--- | :---: | :---: |
| `Nominal_Stiff_1.0` | `18.8%` (`3/16`, `13` bind) | `0.0%` (`0/14`, `14` bind) |
| `Nominal_GravComp_0.35` | `25.0%` (`4/16`, `12` detent) | `0.0%` (`0/14`, `12` detent, `2` bind) |
| `D1_Res_Stiff` *(3 seeds)* | `62.5%` (`30/48`, `18` bind) | **`0.0%`** (`0/42`, `42` bind) |
| **`D2_Res_GravComp_TL`** *(3 seeds)* | **`93.8%`** (`45/48`, `2` bind, `1` detent) | **`21.4%`** (`9/42`, `22` bind, `11` detent) |

Under stiff position control (`D1`), lateral rail misalignment above `10°` jams the prismatic rail immediately (`42/42` bind stalls). Under `GravComp-TL` (`D2`), lateral compliance on `shoulder_pan` (`α_lat ≈ 0.30`) keeps the drawer sliding long enough for the GRU to infer rail yaw and steer into the rail while boosting `α_arm` to clear the detent.

### 4.2 Split OOD Height-Shift Audit (`+15 mm` vs. `-15 mm`) & Unbolted Pinning Physics

| Method | `+15 mm` Height Shift | `-15 mm` Height Shift |
| :--- | :---: | :---: |
| `Nominal_Stiff_1.0` | `40.0%` (`4/10`, `3` shv, `3` tmo) | `20.0%` (`2/10`, `1` shv, `7` tmo) |
| `Nominal_GravComp_0.35` | `10.0%` (`1/10`, `8` shv, `1` tmo) | `10.0%` (`1/10`, `9` shv) |
| `Privileged_Oracle` | `100.0%` (`10/10`, `3.5 mm` disp) | `100.0%` (`10/10`, `6.2 mm` disp) |
| `D1_Res_Stiff` *(3 seeds)* | `33.3%` (`10/30`, `14` shv, `6` tmo) | `36.7%` (`11/30`, `8` shv, `6` bind, `5` tmo) |
| `D2_Res_GravComp_TL` *(3 seeds)* | `16.7%` (`5/30`, `25` shv) | `23.3%` (`7/30`, `22` shv, `1` tmo) |

Two distinct effects are cleanly separated here:
1. **Relative-Kinematics Height Invariance Works:** Unlike `blind_insertion` Phase 6 (where absolute `ee_pose_fk` caused `0%` on `±12 mm` height shifts), removing absolute world `x/y/z` in `blind_drawer` keeps `+15 mm` and `-15 mm` balanced for both `D1` (`33.3%` vs. `36.7%`) and `D2` (`16.7%` vs. `23.3%`).
2. **Physical Bottleneck Behind `D2`'s `47/60` OOD Shoves:** On `280–350 g, μ=0.35` cabinets, table static friction without downward pinning is only `μ·m·g ≈ 1.03 N`, while drawer resistance is `3–7 N`. Opening without shoving (`> 25 mm`) requires pressing **downward** (`-F_z`) onto the handle ledge (`Privileged_Oracle` pins at `1.79°` tilt, `46.8 N` peak force). Because `D2` puts both vertical pinning (`shoulder_lift`, `elbow_flex`, `wrist_flex`) and axial pull under one `α_arm` cap that stayed around `0.36` (`0.86 N·m` margin), `D2` clips downward normal force (`0.37°` tilt vs. `D1` `0.77°`), so `μ(m·g + F_z) < F_pull` and the light cabinet slides. This directly mirrors the Phase 0 baseline gap (`Nominal_GravComp_0.35`: `10%`, `17/20` shoves vs. `Nominal_Stiff_1.0`: `30%`, `4/20` shoves).

---

## 5. Cross-Task Synthesis (`blind_drawer` vs. `blind_insertion`)

| Dimension | `blind_insertion` (Phase 5 & 6) | `blind_drawer` (Phase 1) | Takeaway |
| :--- | :--- | :--- | :--- |
| **Primary Contact Bottleneck** | Rim-contact compliance + unbolted fixture shove | Off-axis rail binding (`±18°`) + sticky detent (`5–9 N`) + unbolted cabinet shove | `GravComp-TL` + relative-kinematics GRU transfers directly to a new manipulation family with zero architecture changes. |
| **Hard Contact-Rich Split** | Split C (`0.5 mm`): `R2_v2` **`46.7%`** vs. `R1_v2` `30.0%` (**+16.7 pp**) | Split A (`bolted_sticky`): `D2` **`60.0%`** vs. `D1` `33.3%` (**+26.7 pp**) | Consistent **+17 to +27 pp** advantage whenever active lateral compliance prevents jamming. |
| **Unbolted In-Dist Split** | Split B (`unbolted`): `R2_v2` **`80.0%`** (`0` shv) vs. `R1_v2` `56.7%` | Split B (`unbolted`): `D2` **`75.6% ± 1.6%`** (`7.2 mm`) vs. `D1` `76.7% ± 18.9%` (`9.8 mm`) | Lower disturbance and dramatically lower seed variance, though unbolted drawer pinning exposes the `α_arm` downward-force trade-off. |
| **Zero-Shot Height Shift** | Phase 6 (`abs z` obs): `0%` on `±12 mm` | Phase 1 (`rel-only` obs): balanced `+15 mm / -15 mm` | Removing absolute `ee_pose_fk` cleanly resolves height-shift memorization. |
| **Next Engineering Lever** | Decouple spiral-brake / relative-only `z` | Decouple `α_arm` pinning torque from lateral compliance `α_lat` (or widen `α_arm` exploration `[0.2, 1.0]`) so `D2` can anchor `300 g` cabinets while keeping `shoulder_pan` soft. | Clear, falsifiable next hypothesis for Phase 2 if desired. |

---

## 6. Verification & Artifacts

- **Unit Tests:** `10/10` passed in `1.43 s` ([test_phase0.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/test_phase0.py) + [test_phase1.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/test_phase1.py)).
- **Results JSON:** [phase1_results.json](file:///home/krisd/physical-ai-systems/projects/blind_drawer/phase1_results.json)
- **Checkpoints (6 frozen policies):** [phase1_checkpoints/](file:///home/krisd/physical-ai-systems/projects/blind_drawer/phase1_checkpoints/) (`D1_Res_Stiff_seed{101,202,303}.npz`, `D2_Res_GravComp_TL_seed{101,202,303}.npz`).
- **Policy & Trainer:** [phase1_policy.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/phase1_policy.py), [phase1_train.py](file:///home/krisd/physical-ai-systems/projects/blind_drawer/phase1_train.py).
