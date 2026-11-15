# SO-101 rigid-bottle baseline

The existing controller does not yet provide a reliable bottle-manipulation baseline. It placed 10/18 bottles in the frozen test, and a common fully closed position target with slower lifting/transport placed 7/18. The earliest common problem is grasp geometry: the bottle moved a median 42.0 mm across the table during jaw closure, compared with its 26 mm diameter. These movements occur before transport and confound any interpretation as load or friction adaptation. Next, inspect jaw/object approach geometry and introduce ordinary visual reobservation of the displaced bottle; verify a stable grasp and placement before testing an adaptive grip policy. This is not evidence for or against active adaptation in general.

The prior camera/tool calibration branch remains closed for its previously tested conditions; see `TRANSFER_RESULTS.md`. That result does not establish this new task's reliability.

## Task and boundary

One run is approach, enclose, close, lift, move, lower, release, and settle. Controllers receive only 320×240 rendered overhead images, joint angles, their own command history, timestamps, nominal FK, and the existing table-marker camera calibration. Joint velocities, true bottle pose/mass/fill/COM/friction, contact forces and disturbance labels are not controller inputs. All bottle conditions use the same opaque green cylinder geometry and color. Images are rendered every fifth action, but the inherited controller localizes the bottle only at the start and does **not** reobserve it after grasping. Overhead vision may reveal lateral displacement if the bottle is visible; occlusion, height ambiguity and a top-only view limit slip and tilt observability. No force sensor is added.

The baseline reuses `nominal.run`, calibration, IK and position feedback, with a single bottle-specific grasp offset (-8, -30) mm selected on development cases. Joint position increments are limited to ±0.08 rad per 0.1 s; translation requests are bounded to 15 mm/action, with the existing fixed wrist-roll and pitch strategy. Jaw open and closed targets are 1.2 and -0.1 rad, held closed for 1.5 s. This is a **position target**, not a measurement or direct command of grip force. A target ramp can be configured and was tested in development. A slower translational target can be configured. The fixed upstream model uses position servos (`kp=998.22`, `kv=2.731`) with actuator output clamped to ±2.94 N·m for all six joints; this is a model setting, not verified physical SO-101 torque, and changing it would be a simulation-only capability here. The comparison targets the minimum jaw position (-0.174533 rad) and halves maximum lift/transport/lower translation; it does not raise this force limit. Position targets and rate limiting have plausible servo counterparts, but hardware behavior is not validated.

## Bottle and contact model

A single upright opaque cylinder approximates a miniature bottle: radius 13 mm, height 40 mm, 12 g tare. Its interior water cylinder has radius 12 mm, height 38 mm and density 1000 kg/m³, with a 1 mm bottom and top allowance. Tare inertia is approximated as a uniform solid cylinder; water is a solid cylinder filling from the bottom. Combine their moments with the parallel-axis theorem. This deliberately does not model a neck, liquid sloshing, deformation, crushing, or changes in visible appearance. It is a 17.19 mL miniature vial, not a typical large water bottle.

| Fill | Water, g | Total mass, g | COM z from geometric center, mm |
|---|---:|---:|---:|
| Empty | 0 | 12.000 | 0 |
| Half | 8.595 | 20.595 | -3.965 |
| Full | 17.191 | 29.191 | 0 |

Inertias are calculated and recorded per episode by `physical_parameters`; all pass positivity and triangle checks. Cylinder collision geometry and object free joint have no attachment, weld or hidden support. A preclosure evaluator reconstruction of all 18 held-out starts found the minimum bottle-to-jaw geometry separation was 23.9 mm with the jaw open. That confirms clearance at those tested poses, not a general measurement of the full aperture. Five of six conditions completed at least one held-out episode with the unchanged ±2.94 N·m model limit; empty–wet did not. Its physical feasibility within these controls therefore remains unresolved, not proven impossible.

Sliding friction is an explicitly **hypothesized**, unvalidated proxy: finger–bottle dry 1.0, wet 0.3. Bottle–table friction is 0.3 in both conditions. The bottle and gripper geoms have equal priority, and MuJoCo takes the elementwise maximum of their coefficients; the evaluator recorded actual finger–bottle contact coefficients of 1.0 and 0.3. The bottle geom has higher priority than the table, making the bottle value govern table contacts. See [MuJoCo contact-parameter combination rules](https://mujoco.readthedocs.io/en/2.3.0/modeling.html#contact-parameters). These values do not validate real plastic, wet fingers or wet-contact physics. MuJoCo uses 5 ms physics steps, implicitfast and elliptic contact; the same two full-bottle development cases gave the same binary dry success/wet failure at 2.5 ms, but this limited sensitivity check does not validate contact forces.

## Frozen experiment and outcomes

After separate box-offset failures and development geometry sweeps, the final 18-case protocol froze three new paired positions, starts and seeds crossed with all six mass/friction conditions. Each method reset all state, used 200 actions/8 rad accumulated arm movement as limits, and had no retries. The comparison used exactly the same cases and budgets. An initial three-episode evaluation was interrupted when the transport-contact audit was found to include expected pickup contact; its records were retained, the audit was fixed, and entirely new held-out cases were frozen before evaluation.

Success was fixed in advance: contact with both jaws after closure, bottle center lifted >35 mm, final destination x/y error <35 mm, settled center within 5 mm of the upright 20 mm resting height, tilt <15°, linear speed <0.015 m/s and angular speed <0.2 rad/s after a 3.5 s release/settling interval. Each entry below is successful placements/3; all failures remain in the records.

| Condition | Existing calibration controller | Fully closed position + slow |
|---|---:|---:|
| Empty, dry | 1 | 0 |
| Empty, wet | 0 | 0 |
| Half, dry | 3 | 1 |
| Half, wet | 2 | 2 |
| Full, dry | 3 | 3 |
| Full, wet | 1 | 1 |
| **Total** | **10/18** | **7/18** |

Seven paired episodes succeeded under both, three only under the original controller, none only under the conservative rule, and eight under neither. This is too small to rank methods generally. Median baseline/comparison simulated time was 8.0/8.8 s and arm movement 2.575/2.539 rad, with 75/83 actions. Median measured controller compute was 6.73/8.15 ms and measured wall time was 3.65/3.89 s; compute excludes physics and rendering. Wall times were obtained on a shared machine and are not a controlled performance benchmark.

Baseline grasp acquisition was 17/18 and lift clearance 15/18; both jaws remained in contact at the end of lift in 15/18. The empty–wet `holdout-a` bottle moved 77.5 mm during closure and was never acquired, while its paired dry case moved 44.1 mm and lifted. In full `holdout-a`, both conditions initially acquired the bottle; the wet bottle was on the table at the lift snapshot whereas the dry bottle remained held. These observations support enclosure and retention failure in these examples, without establishing whether mass, friction, numerical contact or geometry is the ultimate cause. The dry empty bottle also has placement/tipping failures despite lift, so grasp retention alone is insufficient.

Evaluator-only relative movement is maximum displacement of the bottle center in the gripper frame after closure until opening; it is a **relative-motion proxy**, not a tactile slip measurement. After a drop it includes growing separation. Baseline median relative motion was 7.0 mm (range 2.9–120.7 mm); median final x/y error 33.8 mm (14.9–142.9 mm); median peak transport tilt 11.3° (up to 96.8°). Unintended bottle–table contact during transport was recorded at physics-step resolution; `paired.json` labels a drop only if lift clearance had already been achieved and table contact subsequently occurred during transport. This definition may miss airborne drops or contact between action phases.

The evaluator records each finger–bottle normal contact force and sums normal-force×time across contacts at every 5 ms physics step; these are simulation outputs, not controller measurements. Baseline median per-episode maximum individual contact normal was 11.0 N (maximum 18.5 N); median summed normal impulse was 182.0 N·s. These potentially high values, especially with a rigid bottle, should not be interpreted as hardware forces. Absolute actuator output reached the modeled 2.94 N·m limit on several joints. Maximum all-joint target tracking error reached 1.32 rad, dominated by the abrupt commanded jaw closure; arm action saturation and actuator saturation counts are retained per episode. Baseline/comparison had 0/0 prohibited self-collision physics steps and 0/56 joint-limit excursion physics steps (threshold 1 mrad beyond the modeled range). The latter arose with the fully closed target in wet episodes and is a real limitation of that comparison.

## Research implication

Mass and friction could change the appropriate closing and motion profile, but this first task does not establish which changes the best action: closure pushes and uncertain wet feasibility dominate. Ordinary camera images and jaw position/history might reveal lateral movement or an obstructed closing trajectory, but the current policy does not use them after its initial detection. There is no evidence yet that a diagnostic movement is needed. The rigid bottle has no damage penalty, so a realistic grip-versus-crushing tradeoff cannot be evaluated; even the tested maximum-closure-position and slower-motion rule did not trivially solve all cases. No arbitrary damage threshold, force measurement, RL policy, or exploration policy is introduced.

## Artifacts and reproduction

Frozen baseline: `results/so101_runs/20261115T182403.955615Z_3a54587f690f_bottle_13b8f712/`; comparator: `results/so101_runs/20261115T182532.332957Z_3a54587f690f_bottle_1b3107b2/`. Both contain six condition videos, source snapshots, dependency/revision/config metadata, summaries and full episode traces (also compacted as `episodes_full.json.gz`). Paired evaluator-only metrics and a plot are in `results/so101_runs/20261115T182707.511772Z_3a54587f690f_bottle_analysis_b29c40df/`. Development, interrupted, aperture and timestep diagnostic runs remain separate. Source snapshots retain the earlier development version before the 1 mm interior-bottom correction; frozen runs use the corrected centered full-water geometry.

From the project directory:

```bash
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/bottle_evaluation.py --config so101/bottle_frozen_baseline.json
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/bottle_evaluation.py --config so101/bottle_frozen_conservative.json
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/bottle_report.py <baseline-run-directory> <comparison-run-directory>
.venv-so101/bin/python so101/bottle_plot.py <analysis-run-directory>
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/bottle_aperture.py <baseline-run-directory>
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/bottle_evaluation.py --config so101/bottle_timestep_5ms.json
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/bottle_evaluation.py --config so101/bottle_timestep_2_5ms.json
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python -m unittest discover -s so101 -p 'test*.py'
```

Each writer creates a unique run directory; the plotting file is created exclusively. No physical system, paid compute, or policy training was used.
