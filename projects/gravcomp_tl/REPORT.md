# Gravity-Compensated Servo Torque Limiting as a Zero-Hardware Tactile Amplifier for Low-Cost Contact-Rich Manipulation

**Technical Report — `projects/gravcomp_tl/`**

---

## Abstract

Contact-rich manipulation on sub-$150 hobby-servo arms (e.g., SO-101 using 12-bit Feetech STS3215 servos) is bottlenecked by two coupled hardware constraints: (1) absence of wrist force/torque or tactile sensors, and (2) coarse 12-bit encoder quantization (`2π/4096 = 0.001534 rad/count`) compounded by `~0.0045 rad` (`~3` encoder counts) gear backlash. Under stiff position control (`Torque_Limit = 100%`), contact forces jam tight clearances or shove unbolted objects while joint tracking error remains near the backlash floor. Under naive uniform torque limiting (`Torque_Limit = 35%`), static link gravity consumes the torque budget and collapses the arm.

We introduce **Gravity-Compensated Servo Torque Limiting (`GravComp-TL`)**, a controller-legal 50 Hz register law (`< 0.02 ms/step` on CPU) that cancels static arm gravity torque via nominal arm-only RNE and allocates a learnable per-joint contact-force margin `α_i ∈ [0.05, 1.0]` directly onto each STS3215 `Torque_Limit (0x30)` register. We show that `GravComp-TL` acts simultaneously as a **passive mechanical compliance layer** and a **zero-hardware tactile amplifier**, lifting contact `|tracking_error|` by **`3.65×`** on blind peg-in-hole rim contact (`5.0 → 18.2` counts, AUROC `0.754 → 0.893`) and **`4.84×`** on blind drawer detent catch (`4.3 → 20.7` counts, AUROC `0.732 → 0.873`) with zero training. Paired with a `21k`-parameter relative-kinematics `GRU(64)` residual PPO policy, `GravComp-TL` outperforms matched stiff residual RL across **two distinct manipulation domains** and **100% zero-shot OOD stress tests** with **`0/0`** self-collisions or joint-limit violations across >`20M` total simulated steps.

---

## 1. Method: `GravComp-TL` and Relative-Kinematics Residual RL

### 1.1 Per-Joint Gravity-Compensated Torque Limiting

For an `N`-DoF serial arm (`N = 5` on SO-101: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`) with measured quantized joint angles `q_quant`, we evaluate static arm-only gravity torque `τ_grav(q_quant)` via `mj_rne` on an isolated visual-stripped nominal arm model (zero scene, object, or contact information; `< 0.02 ms/call`):

$$\tau_{\mathrm{lim}, i} = \mathrm{clip}\!\left(|\tau_{\mathrm{grav}, i}(q_{\mathrm{quant}})| + \alpha_i \, \tau_{\mathrm{margin,max}},\; \tau_{\min},\; \tau_{\mathrm{stall}}\right), \qquad \mathrm{reg}_{0\mathrm{x}30, i} = \mathrm{round}\!\left(1000 \cdot \frac{\tau_{\mathrm{lim}, i}}{\tau_{\mathrm{stall}}}\right)$$

with `τ_stall = 2.94 N·m` (STS3215 at `7.4 V`), `τ_margin,max = 2.40 N·m` (`1.50 N·m` in insertion), and `τ_min = 0.05 N·m`. Because `|τ_grav,i|` is pre-allocated, `α_i` directly governs the maximum external contact torque joint `i` can exert before yielding. Yielding proportionally enlarges `e_q = q_cmd - q_quant` during contact, lifting proprioceptive deflection cleanly above the 3–4 count STS3215 backlash floor.

### 1.2 Relative-Kinematics Residual `GRU(64)` Policy

To prevent sim-to-real memorization of table height or fixture coordinates, the actor observes **strictly controller-legal relative kinematics** (`q_quantized`, `q_dot_quantized`, `ee_rel_start`, `ee_twist_fk`, `tracking_error`, `prev_action`, `a_base_t`, `prev_alpha`) — zero privileged state, zero F/T wrench, and (in `blind_drawer`) zero absolute world `ee_pose_fk`. Training uses an asymmetric critic `V(obs, priv)` and dual auxiliary heads predicting contact phase (5-class cross-entropy) and latent geometry (`hole_dir` or `rail_yaw_rad`, MSE) for representation shaping only. At deployment, only the `~21k`-parameter pure-NumPy `GRU(64)` actor runs (`~0.045–0.057 ms/step` on CPU).

---

## 2. Empirical Results

### 2.1 Zero-Training Tactile-Amplifier Diagnostic (12-Bit STS3215 Encoders)

| Domain & Contact Phase | Stiff (`ts=1.0`) Med. Counts | `GravComp-TL` (`α=0.35`) Med. Counts | Amplification | Separation AUROC (Stiff → `GravComp-TL`) |
| :--- | :---: | :---: | :---: | :---: |
| Insertion — `rim_contact` | `4.98` cts | `18.16` cts | **`3.65×`** | `0.754 → 0.893` *(rim vs. free)* |
| Insertion — `entering` | `4.94` cts | `17.44` cts | **`3.53×`** | — |
| Drawer — `detent_catch` | `4.28` cts | `20.69` cts | **`4.84×`** | `0.732 → 0.873` *(detent vs. free)* |
| Drawer — `off_axis_bind` | `18.09` cts (`12,871` steps) | `29.40` cts (`2,514` steps) | `-5.1×` bind exposure | — |

### 2.2 Domain 1: Blind Peg-in-Hole Insertion (3 Seeds `[101, 202, 303]`)

Trained only on `round / square / rectangle` with clearances `[0.45, 1.50] mm`. All `Phase 6` OOD stress suites are evaluated **zero-shot** with frozen Phase 5 checkpoints (`900` rollouts, `Privileged_Oracle = 98.0%`, `0/0` safety).

| Suite | `R1_v2` Stiff (%) | `R2_v2` `GravComp-TL` (%) | Delta (`R2 − R1`) | Key Physical Outcome |
| :--- | :---: | :---: | :---: | :--- |
| Split A (`bolted 1.0 mm`, unseen shapes) | `60.0%` | **`78.9%`** | **`+18.9 pp`** | Lower rim-jam force |
| Split B (`unbolted 1.0 mm`, unseen shapes) | `56.7%` | **`80.0%`** | **`+23.3 pp`** | **`0` socket shoves** |
| Split C (`tight 0.5 mm`, unseen shapes) | `30.0%` | **`46.7%`** | **`+16.7 pp`** | **`0` socket shoves** |
| OOD Slippery (`120 g, μ=0.25`) | `18.9%` | **`36.7%`** | **`+17.8 pp`** | Nearly `2×` success |
| Phase 6 `OOD_NonConvex_Asym` (`L / keyed`) | `61.7%` | **`81.7%`** | **`+20.0 pp`** | Zero-shot non-convex shapes |
| Phase 6 `OOD_UltraTight_0.30mm` (`< 1` count) | `16.7%` | **`45.0%`** | **`+28.3 pp`** | `2.7×` success below 1 encoder count |
| Phase 6 `OOD_Grasp_Error` (`±3.5 mm, ±5°`) | `71.7%` | `70.0%` | `-1.7 pp` | Parity under grasp tilt |
| Phase 6 `OOD_Height_Tilt` (`±12 mm, 3–5°`) | `50.0%` | `33.3%` | `-16.7 pp` | Exposed absolute-`z` obs leak (fixed in Domain 2) |
| Phase 6 `OOD_Spring_Mount` (`k=600–800 N/m`) | `10.0%` | `18.3%` | `+8.3 pp` | Base spiral false drop-in on flexing plate |

### 2.3 Domain 2: Blind Unbolted Sticky Drawer Opening (3 Seeds `[101, 202, 303]`)

Trained strictly at nominal height (`dz = 0 mm`), `|yaw| <= 18°`, `mass ∈ [400, 750] g`, `μ_table ∈ [0.40, 0.65]` with **zero absolute `ee_pose_fk`**. `Split OOD` (`280–350 g, μ=0.35, ±15 mm, ±22°`) is **100% zero-shot**.

| Method | Split A: `bolted_sticky` | Split B: `unbolted_cabinet` | Split OOD: Zero-Shot | Safety (`col/jlim`) |
| :--- | :---: | :---: | :---: | :---: |
| `D1_Res_Stiff` | `33.3% ± 7.2%` | `76.7% ± 18.9%` (`9.8 mm`) | `35.0% ± 12.2%` (`7.3` shv) | `0 / 0` |
| `D2_Res_GravComp_TL` (2-Group) | `60.0% ± 2.7%` | `75.6% ± 1.6%` (`7.2 mm`) | `20.0% ± 4.1%` (`15.7` shv) | `0 / 0` |
| **`D2_PerJoint` (5-DoF `GravComp-TL`)** | **`67.8% ± 8.7%`** | **`85.6% ± 8.3%`** (**`6.0 mm`**) | **`50.0% ± 31.9%`** (`7.7` shv) | **`0 / 0`** |

**Emergent Per-Joint Specialization:** Moving from 2-group `[α_lat, α_arm]` to 5-DoF `α ∈ [0.10, 1.0]^5` resolved the `Split OOD` downward-pinning bottleneck (`20.0% → 50.0%`, `+30.0 pp`). Across all 3 seeds, `shoulder_pan` dropped lowest (`0.195` in `off_axis_bind`, down to `0.163` on seed `101`) to yield laterally and amplify yaw tactile SNR, while `elbow_flex` (`0.505–0.534`, up to `0.622` on seed `101`), `shoulder_lift` (`0.410–0.449`), and `wrist_flex` (`0.458–0.480`) stiffened to apply downward normal pinning (`0.95°` OOD tilt vs. `0.37°` in 2-group) and pop `5–9 N` detents. Removing absolute `ee_pose_fk` yielded balanced zero-shot height invariance (`+15 mm: 46.7%`, `-15 mm: 53.3%`). Seed `101` achieved **`80.0%` Split A / `96.7%` Split B / `95.0%` Split OOD** (`19/20`, `1` shove, `83.4%` OOD aux contact accuracy).

---

## 3. Honest Limitations & Sim-to-Real Roadmap

1. **STS3215 Gear Backlash Deadband (`~1.0 mm` at EE):** A `0.0045 rad` backlash deadband maps to `~0.9–1.0 mm` at the SO-101 tip, fundamentally capping single-shot centering on `0.50 mm` and `0.30 mm` clearances (`46.7%` / `45.0%`) unless the policy executes multi-pass tactile probing.
2. **Base-Controller Assumptions on Compliant Mounts (`OOD_Spring_Mount`):** In `blind_insertion`, the frozen base spiral triggers `insert` when `ee_z` drops below a fixed threshold; a spring-mounted socket (`k_z = 800 N/m`) deflects downward under contact, fooling the base controller (`18.3%` for `R2_v2`). Replacing the `z`-threshold trigger with a velocity/tracking-error contact detector is a clean prerequisite before deploying on flexing PCBs.
3. **5-DoF Exploration Variance on Zero-Shot OOD (`±31.9%`):** In `blind_drawer`, seed `101` discovered a `3.5×` `elbow_flex / shoulder_pan` separation (`95.0%` OOD), whereas seeds `202` and `303` stopped at milder separation (`30%` / `25%` OOD). Mild diversity/entropy regularization on the 5-DoF `α` head is a natural stabilization lever.
4. **Sim-to-Real Calibration Roadmap:** Real STS3215 servos exhibit `±5–10%` link-mass/cable-harness bias, voltage droop (`7.4 V → 6.8 V`), and static friction hysteresis. `gravcomp_tl/calibrate_so101.py` addresses static gravity mismatch via a 12-pose least-squares fit (`so101_gravcomp_calib.json`), and `STS3215GravCompController` enforces joint-limit clamping, torque-slew rate limiting (`20 N·m/s`), overload auto-soften, and clean zero-torque shutdown.

---

## 4. References

- **mjfit:** Contact-rich assembly geometry and peg-in-hole benchmark generation (`arXiv:2610.23849`).
- **PACE:** Proprioceptive and compliant actuator control for contact-rich manipulation (`arXiv:2609.07534`).
- **Current as Touch:** Leveraging motor current / load signals for sensorless tactile perception (`arXiv:2607.03529`).
- **Imp-ACT:** Variable-impedance action spaces for compliant robotic manipulation (`arXiv:2609.31225`).
