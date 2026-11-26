# Phase 0 Gate Verification — Blind Unbolted Sticky/Latched Drawer Opening (`projects/blind_drawer/`)

**Gate Status:** **PASS** (Ready for Phase 1 RL review — ZERO RL trained yet).
**Artifacts:**
- Core environment & GravComp-TL helper: [`env.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/env.py), [`gravcomp.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/gravcomp.py)
- Deterministic benchmarks & reference controllers: [`benchmarks.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/benchmarks.py), [`controllers.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/controllers.py)
- Gate runner, unit tests, & raw JSON: [`phase0_gate.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/phase0_gate.py), [`test_phase0.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/test_phase0.py), [`phase0_results.json`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/phase0_results.json)
- `projects/blind_insertion/` remains 100% untouched.

---

## 1. Environment & Observation-Boundary Verification

- **Physics & Servo Fidelity (Matched to Real Feetech STS3215 SO-101):**
  - `1 kHz` MuJoCo physics (`timestep = 0.001 s`), `50 Hz` policy (`20` substeps, `DT = 0.02 s`), 12-bit STS3215 quantization (`2π/4096 = 0.001534 rad`), `0.0045 rad` stateful gear backlash hysteresis, and `GravComp-TL` (`lim_i = clip(|tau_grav_i(q_quant)| + alpha_i * 2.40, 0.05, 2.94)`).
  - Native SO-101 jaws (`fixed_jaw` + `moving_jaw_so101_v1`) start closed cleanly around a `9 mm` horizontal drawer pull bar (no welded constraints; realistic jaw friction `mu = 0.85`).
  - Prismatic drawer (`travel ∈ [0, 90] mm`, success `>= 70 mm`), real Coulomb off-axis rail-binding friction computed from exact MuJoCo internal prismatic joint reaction force (`mj_rnePostConstraint` → `cfrc_int[drawer]` projected onto rail lateral axis), and smooth raised-cosine physical detent catch at unknown `d_detent ∈ [10, 40] mm`.
  - Unbolted tabletop cabinet fails if base slides `> 25 mm`, tilts `> 10°`, or handle slips out of jaws (`> 28 mm`).
- **Strict Relative-Kinematics `obs` Boundary (Baking in Phase 6 Lessons):**
  - `obs` contains **ONLY 6 controller-legal keys**: `q_quantized`, `q_dot_quantized`, `ee_rel_start` (`ee_pos_fk - ee_start_pos_fk`), `ee_twist_fk`, `tracking_error` (`q_cmd - q_quantized`), `prev_action`.
  - **ZERO absolute `ee_pose_fk` / `ee_pos_fk`**, ZERO rail yaw, ZERO detent location/force, ZERO cabinet mass/friction, ZERO F/T wrench (enforced in `test_phase0.py`).
  - Training-only `info['privileged']` provides `drawer_travel_mm`, `rail_yaw_rad`, `cabinet_disp_mm`, `cabinet_tilt_deg`, `peak_contact_force_N`, and `contact_class ∈ {free_slide, off_axis_bind, detent_catch, handle_slip, opened}`.

---

## 2. Reference Controllers Across All 3 Deterministic Held-Out Splits

- **Split A (`bolted_sticky`, n=30):** Bolted, `θ_rail ∈ [-18°, +18°]`, `f_rail ∈ [2.5, 5.5] N`, `F_detent ∈ [5.0, 9.0] N`.
- **Split B (`unbolted_cabinet`, n=30):** Unbolted (`400–750 g`, `μ_table ∈ [0.40, 0.65]`), `θ_rail ∈ [-18°, +18°]`, `F_detent ∈ [3.0, 7.0] N`.
- **Split OOD (`ood_light_height_angle`, n=20):** Light unbolted (`280–350 g`, `μ_table = 0.35`), **unseen cabinet height shift `±15 mm`**, steeper yaw `±22°`.

| Controller | Split A (`bolted_sticky`, n=30) | Split B (`unbolted_cabinet`, n=30) | Split OOD (`light+height+yaw`, n=20) | Safety (`col / jlim`) |
| :--- | :---: | :---: | :---: | :---: |
| **`Nominal_Stiff_1.0`** (`ts=1.0`) | **`10.0%`** (`3/30`, `27` bind stalls, `25.2 N`) | `70.0%` (`21/30`, `2` shv, `7` partial, `12.3 mm` disp, `23.5 N`) | **`30.0%`** (`6/20`, `4` shv, `10` partial, `21.3 mm` disp, `20.7 N`) | `0 / 0` |
| **`Nominal_RawLowTorque_0.35`** (`ts=0.35`) | `10.0%` (`3/30`, `16` bind, `11` detent, `22.4 N`) | `36.7%` (`11/30`, **`17` shv**, `25.1 mm` disp, `24.1 N`) | `5.0%` (`1/20`, **`19` shv**, `25.3 mm` disp, `21.7 N`) | `0 / 0` |
| **`Nominal_GravComp_0.35`** (`α=0.35`) | `13.3%` (`4/30`, **`2` bind**, `24` detent, `23.3 N`) | `56.7%` (`17/30`, `13` shv, `17.4 mm` disp, `24.2 N`) | `10.0%` (`2/20`, `17` shv, `25.3 mm` disp, `21.9 N`) | `0 / 0` |
| **`Privileged_Oracle`** (Upper Bound) | **`100.0%`** (`30/30`, `0.38 s`, `49.5 N`) | **`100.0%`** (`30/30`, `1.74 s`, **`0.5 mm`** disp, `0` shv, `39.5 N`) | **`100.0%`** (`20/20`, `2.41 s`, **`4.6 mm`** disp, `0` shv, `46.8 N`) | `0 / 0` |

### Zero-Training Physics Findings
1. **Oracle Solvability Confirmed (`100.0%` on all 3 splits, `0/0` safety):** Proves every split (including `±15 mm` height shifts and `±22°` steep angles on `280–350 g, μ=0.35` unbolted cabinets) is cleanly solvable within SO-101's real `2.94 N·m` STS3215 torque envelope.
2. **Stiff Straight-Back Pull Fails Severely on Binding & OOD:** `Nominal_Stiff_1.0` collapses to **`10.0%`** on Split A (`27/30` `off_axis_bind_stall`, well below the `<= 40%` gate) and **`30.0%`** on Split OOD (`21.3 mm` median cabinet drag).
3. **`GravComp-TL (α=0.35)` vs. Raw Low Torque (`ts=0.35`) & Stiff (`ts=1.0`):**
   - On **Split A (`bolted_sticky`)**, fixed `GravComp-TL (α=0.35)` slashes `off_axis_bind_stall` by **`93%`** (`27/30 → 2/30`), converting hard off-axis lockups into clean `detent_stall` (`24/30`) that require dynamic axial torque ramp (`α_arm ↑`) to pop through `5–9 N` detents.
   - On **Split B (`unbolted_cabinet`)**, `GravComp-TL (α=0.35)` eliminates gravity sag and beats raw `ts=0.35` by **`+20.0 pp`** (`56.7%` vs. `36.7%`, reducing median cabinet displacement `25.1 → 17.4 mm`). However, fixed scalar `α = 0.35` caps vertical stabilizing force on unbolted cabinets (`13` shoves vs. `2` for Stiff), clearly motivating **2-group dynamic `[α_lat, α_arm]`** + closed-loop yaw steering in Phase 1 RL.

---

## 3. Zero-Training Cross-Task Tactile-Amplifier Diagnostic (80 Matched Trials A + B + OOD)

Measured `|tracking_error|` (5-DoF L2 and max-joint) in `rad` and **12-bit STS3215 encoder counts** (`1 count = 2π/4096 = 0.001534 rad`) across all 80 matched trials under `Nominal_Stiff_1.0` vs. `Nominal_GravComp_0.35`:

| Contact Class | Steps (`Stiff / GC`) | Stiff Med L2 (`rad / cts`, `>5 ct`) | `GravComp α=0.35` Med L2 (`rad / cts`, `>5 ct`) | Stiff Med Max-J (`cts`, `>5 ct`) | `GravComp` Med Max-J (`cts`, `>5 ct`) | Med L2 Ratio (`GC / Stiff`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`free_slide`** | `5,110 / 4,475` | `0.00533 / 3.5 cts` (`5.7%`) | `0.00943 / 6.2 cts` (`77.0%`) | `2.7 cts` (`2.4%`) | `5.9 cts` (`71.3%`) | **`1.77×`** |
| **`off_axis_bind`** | `12,871 / 2,514` | `0.02775 / 18.1 cts` (`75.2%`) | `0.02935 / 19.1 cts` (`97.5%`) | `14.6 cts` (`72.0%`) | `15.3 cts` (`96.5%`) | **`1.06×`** *(`5.1×` fewer bind steps)* |
| **`detent_catch`** | `344 / 7,385` | `0.00656 / 4.3 cts` (`24.4%`) | **`0.03173 / 20.7 cts`** (**`97.3%`**) | `3.8 cts` (`12.5%`) | **`16.4 cts`** (**`96.1%`**) | **`4.84×`** |

### Separability AUROC (`|tracking_error|` L2 Alone, Zero Training)
- **`AUROC(detent_catch vs. free_slide)`:** Stiff `0.7317` → **`GravComp-TL α=0.35` `0.8732` (`+0.142`)**
- **`AUROC(off_axis_bind vs. free_slide)`:** Stiff `0.9345` vs. `GravComp-TL α=0.35` `0.8294`

> **Cross-Task Tactile-Amplifier Confirmed:** Just as `GravComp-TL` amplified `rim_contact` by `3.65×` (`5.0 → 18.2 counts`) on peg-in-hole insertion, on blind drawer opening it amplifies **`detent_catch` by `4.84×` (`4.3 → 20.7 counts`, `> 5 counts`: `24.4% → 97.3%`)**, lifts `off_axis_bind` `> 5 counts` detection to **`97.5%`** while cutting off-axis binding exposure **`5.1×` (`12,871 → 2,514` steps)**, and boosts `detent vs. free_slide` AUROC by **`+0.142`**.

---

## 4. Throughput, Bounded RAM, & Unit-Test Audit

- **Headless Throughput (8 CPU Workers, `render=False`, warm cache):** **`1,629 steps/s`** (bounded per-worker cache of `<= 4` lean models, `nbuffer < 4 MB` each).
- **Total RSS Memory (Parent + All 8 Workers):** **`0.84 GB → 0.84 GB`** (strictly flat, `< 2 GB` cap with `> 1.1 GB` headroom, `0 B` swap).
- **Safety:** **`0` self-collision steps, `0` joint-limit violations** across all `320` gate rollouts (`+240` warm-benchmark rollouts).
- **Unit Tests ([`test_phase0.py`](file:///home/krisd/physical-ai-systems/projects/blind_drawer/test_phase0.py)):** **`5/5` passed in `1.36 s`** (zero absolute `ee_pose_fk` / zero privileged keys in `obs`, `GravComp-TL` invariance & `< 0.1 ms/call`, backlash & deterministic replay, torque clamping, `Oracle` 100% across bolted/unbolted/OOD `±15 mm`).
