# Camera/tool correction transfer

## Branch decision

Retain the existing calibration controller and close this camera/tool ambiguity branch **for this simulated task and these bounded offsets**. It placed 36/36 objects after settling, including 18/18 with table references unavailable. No passive estimator or extra probe was implemented or tested: there was no observed task failure requiring one. These data do not measure the benefit of a probe or establish that probing is never useful. One placement ended 32.43 mm from the destination, only 2.57 mm inside the 35 mm success radius; this is a limited margin, and the small deterministic set does not establish robust generalization.

## Model and observability

The known camera model fixes orientation, focal length and height; only its world x/y translation is estimated. Visible table fiducials provide two redundant translation estimates. Otherwise, one cyan gripper marker is localized in the image and its expected position and height are computed from nominal forward kinematics and measured joint angles. Its 2D observation cannot independently identify camera x/y and an arbitrary three-component tool displacement. Marker self-calibration assumes the encoder-to-marker kinematics and marker height are accurate. It may return a useful action correction even when its camera estimate is wrong.

The simulated disturbance translates the entire gripper body, jaws, and attached marker together in the **parent wrist frame**, before the wrist-roll joint; nominal geometry assumes zero displacement. Under this exact rigid model the marker and grasp site have the same world displacement at a fixed configuration. The marker therefore provides indirect evidence about physical grasp-point error only with that rigid relationship, camera model and nominal kinematics assumed; it cannot prove real jaw contact geometry. Wrist roll alone does not rotate this imposed parent-frame displacement. Shoulder and wrist-flex motion do rotate its parent frame. Pure rotation about an offset axis leaves that offset unchanged; under overhead perspective, an offset parallel to a viewing ray is invisible in a single image. A height offset may couple with image x/y depending on camera geometry.

An evaluator-only local sensitivity analysis stacks camera x/y and parent-frame offset x/y/z projection derivatives for one nominal held-out trajectory. One marker observation has rank 2 for these five unknowns. Six idealized sampled configurations have rank 5 (singular values, divided by focal length: 7.720, 7.701, 1.228, 1.056, 0.117), but the weakest direction is much less sensitive. This is a **geometric possibility**, not a demonstrated estimator: it assumes the marker can be detected accurately at every sampled pose and all other camera and geometry parameters are known. Occlusion, pixel quantization, noise, motion, and extra unknowns can destroy practical identifiability. Existing control does not consume a marker history.

## Orientation coverage and transfer scoring

Retrospective analysis of the previously recorded integrated-controller trajectories found up to 40.7° of gripper-body rotation from calibration by reach, and 53.5° by lowering. The frozen fresh trials reached 43.7° by reach and 56.8° during transport; wrist-roll itself remains fixed by the controller. These are full gripper-body rotation angles, not wrist-roll angles. Full recorded trajectories and stage summaries retain the distinction.

Evaluator-only diagnostics reconstruct the true and nominal grasp-site positions at the same recorded joint configuration using separate MuJoCo models. This avoids comparing MuJoCo's pre-integration cached site position with its post-integration joint angles. At jaw closure, grasp-relative prediction error is the difference between (nominal grasp site minus visually estimated object position) and (actual grasp site minus actual object position), in tabletop x/y. It combines the effects relevant to acting, rather than treating camera-parameter error as task error. Camera error compares estimated translation with evaluator-only true translation. Actual alignment uses the logged site-minus-object vector; its deliberate nominal offset is approximately (-11.61, -12.40) mm, so zero is not the desired alignment. The logged site pose is the original cached MuJoCo observation and may lag joint state by one 2 ms physics step. The 3D rigid tool displacement itself is at most 5 mm in this protocol.

| Table references | Success | Closure prediction error, mm range | Camera translation error, mm range | Final placement error, mm range | Closure site/object x/y distance, mm range |
|---|---:|---:|---:|---:|---:|
| Visible | 18/18 | 0.36–6.15 | 1.56–1.81 | 3.40–32.43 | 13.40–21.25 |
| Unavailable | 18/18 | 0.67–5.12 | 1.48–5.60 | 1.13–13.54 | 15.67–20.41 |

Each of the nine camera/tool combinations contributed two paired initial configurations to each reference condition. All 36 episodes had successful two-jaw contact, lifting, and settled placement; no retries, prohibited self-collision physics steps, or joint-limit excursion physics steps were recorded. The calibration controller has no motion-adaptation updates; each episode performs one initial accepted calibration and resets memory. Per-episode calibration decisions, failures, positions, timing, actions, and tracking errors are retained in the machine-readable records.

For visible/unavailable references respectively, median actions were 80/77; median accumulated five-joint movement 2.525/2.537 rad; median simulated time 8.5/8.2 s; median measured controller compute time 0.0079/0.0079 s; and median episode wall time 3.19/3.21 s on this machine. Compute time covers explicitly timed localization, controller and estimator work, not MuJoCo physics or rendering. Simulated time includes reset settling, gripper closure, and release settling. Movement excludes gripper motion, so compare it only under this shared protocol. Total budget was 200 actions and 8 rad of accumulated arm motion per episode; the same limits would have included a probe had one been run.

## Protocol and artifacts

Four development episodes passed, including two marker-only tool-offset episodes; they did not enter evaluation. The frozen test contains two new object positions/initial pan angles and seeds 1901/1902, crossed with camera shifts (0,0), (+9,-9), (-7,+9) mm and parent-frame tool shifts (0,0,0), (+4,0,0), (-5,0,0) mm, with table references visible or unavailable. Disturbance magnitudes were not enlarged. Starts and disturbances are paired across reference conditions; no memory carries between episodes. Every episode is retained, including any failures; only the existing calibration controller was evaluated. The task uses the existing fixed-roll grasp strategy, so these results do not establish transfer to arbitrary wrist orientations, a different grasp, different geometry, noisy perception, or physical hardware.

Run directory: `results/so101_runs/20261114T192742.266058Z_62ccfe7bfde3_integrated_7000928f/`. `metadata.json` and `source_snapshot.json.gz` record the exact task source, dirty state, configuration, dependencies and revision at execution; `episodes_full.json.gz` preserves all traces. `transfer_metrics.json`, `observability.json`, `summary.json`, and `transfer_scatter.svg` provide derived analysis. Four GIFs show representative image detections and evaluator-projected targets/tips; overlays use simulator truth only for display, never for control. The retrospective orientation run is `results/so101_runs/20261114T193016.138991Z_62ccfe7bfde3_orientation_coverage_cfc3d8b1/`.

Reproduce from the project directory in the existing environment:

```bash
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/orientation_coverage.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/integrated_evaluation.py --config so101/transfer_frozen.json
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/transfer_analysis.py <new-run-directory>
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/observability.py <new-run-directory>
.venv-so101/bin/python so101/transfer_plot.py <new-run-directory>
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python -m unittest discover -s so101 -p 'test*.py'
```

Each evaluation creates a unique directory; analysis files are created exclusively and do not overwrite existing evidence. Repeating may vary wall/compute time and environment-dependent physics/rendering details. The next action is to retain the current calibration on this tested task; revisit information gathering only if a separately motivated task or sensing condition produces a repeatable consequential failure. No general conclusion about active adaptation follows.
