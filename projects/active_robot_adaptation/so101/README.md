# SO-101 tabletop simulation checkpoint

This is a simulated, single-object manipulation pilot, not a calibrated digital twin or a physical-robot result. The 2D baseline is preserved separately in commit `76be36a` and `../PLANAR_CHECKPOINT.md`.

## Model and setup

The unmodified SO-101 MJCF and meshes come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101), revision `370c43965323c0944b433b202aac5058b5f907a9`, Apache-2.0. Only this model directory is fetched into ignored `.so101-model/`; it is not committed. `scene.xml` adds a plane, fixed camera, lightweight box, visible destination, and two magenta calibration marks. It does not modify or disable upstream collisions. The author model uses position-served joints and a hinged gripper in radians; it does not claim that its servo gains match physical LeRobot gains. The separate upstream SO-101 documentation notes its gripper convention differs from normalized LeRobot commands; no hardware mapping is assumed here.

The [reported self-collision](https://github.com/google-deepmind/mujoco_menagerie/issues/325) remains present in the pinned model: with other joints at zero, shoulder/gripper penetration appears at elbow 1.55 rad, and shoulder/lower-arm and shoulder/wrist penetration at 1.69 rad. We did not alter collision geometry. The 0.1 s sampled-state trace audit found zero such shoulder–arm contacts in the tested episodes; this does not exclude contacts between samples. `results/so101/model_checks.json` contains the scan. Actual hardware clearances remain unverified.

From the project directory, with Python 3.12 and a working OpenGL/EGL runtime:

```sh
python3.12 -m venv .venv-so101
.venv-so101/bin/python -m pip install -r requirements-so101.lock.txt
sh so101/fetch_model.sh
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/checks.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python -m unittest discover -s so101 -p 'test_*.py' -v
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/pilot.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/compare.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/compare.py --transfer
```

These evaluation commands overwrite corresponding results. The optional graphical viewer is `PYTHONPATH=so101 .venv-so101/bin/python so101/view.py`; it requires a working display and was not interactively verified. `so101/plot_results.py` uses the existing 2D `.venv` and Matplotlib to regenerate the comparison plot.

## Interface and measurement

The calibrated base/world frame uses metres and radians, z upward. A 320×240 fixed overhead RGB image localizes a green box by color. A known fixed object top height projects the detected pixels onto a plane; there is no depth estimate, general recognition, changing object height or occlusion recovery. Two fixed magenta table markers have declared known world coordinates. All methods receive the same image, simulated encoder-like joint angles, previous commands, and time; only the recalibration method computes a translation from the visible markers. The controller uses a declared nominal FK model, fixed gripper grasp offset and approximate pitch constraint. Its task-facing command is a Cartesian displacement bounded to 0.015 m per 0.1 simulated seconds, plus a gripper target bounded to the model's joint range. The actuator wrapper clips each sent joint increment to ±0.08 rad. Arbitrary six-axis tool poses are not assumed. The intermediate joint-level wrapper is an implementation detail; `rl_interface.py` provides a Cartesian reset/step smoke-test interface without a training reward.

The evaluator sets the initial object position and disturbances but does not pass them, camera extrinsics, or change labels to control decisions. Successful execution requires recorded contact with both jaws after closure, an object lift exceeding 0.035 m, then an object center within 0.035 m of the destination and below 0.04 m after 3.5 s of release/settling. There is no scripted attachment and no oracle-driven correction. Object truth is used for scoring only. The controller executes its planned sequence without branching on truth-based grasp/lift measurements; all failures remain in results. Simulation time includes 0.5 s initial settling; per-episode action count and movement start afterwards. Recovery time is time from reset to a successful settled placement; failures have null recovery time. Between-trial changes are not within-episode recovery measurements. No calibration movement or retry occurs. Controller computation sums timed segmentation, recalibration, FK, inversion and adaptation; it excludes rendering and other program overhead. Tracking error integrates controller FK-to-waypoint error while moving, not camera or object localization error.

Disturbances are separate and applied before reset, without notification: an overhead camera translation of `[+0.012, -0.008]` m with stale nominal calibration, and software command-increment gains `[1, 0.7, 0.8, 1, 1]` applied between the controller and position servo targets. The latter is not a modeled physical actuator fault. Equal cases, initial poses, object positions, action bounds and sensor access are used. Every episode resets model/adaptation memory. Random seeds are recorded; current experiments use zero camera noise, so seed is inactive in their dynamics. Seeded image-noise repeatability is tested separately.

The online Jacobian estimates command-to-observed-joint response via RLS and composes it with nominal FK. The global transfer estimates a 3D FK-predicted-to-observed displacement map; the GP variant adds a bounded-history residual. These are independent, simplified FLAIR-style arm ports, not original FLAIR code or faithful robot reproductions. They omit FLAIR's delay matcher, dataset balancing and change-detection/reset logic. Both use encoder/FK response and do not estimate camera extrinsics. The preexisting 2D Tinygrad policy is not used here, and no SO-101 policy was trained.

## Pilot evidence

The readiness gate was at least 5/6 development successes across multiple starts; the final development sweep succeeded 6/6. The first sweep failed 2/6 and informed improvements to waypoint precision and placement alignment, so development success is not independent evaluation. Four separate object/start cases and three conditions were frozen before the initial comparison. That comparison was repeated after the transfer port without changing its methods; all outcomes and action counts matched. The global and GP ablation was specified only after inspecting the first results and reused the same cases; it is not an independent final test.

| Condition | Fixed nominal | Marker recalibration | Online Jacobian | Global* | Global + GP* |
|---|---:|---:|---:|---:|---:|
| No change | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| Camera translation | 0/4 | 4/4 | 0/4 | 0/4 | 0/4 |
| Joint-command gain | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |

\* Follow-up on previously inspected cases. In the camera condition, all failed methods missed the grasp. The camera recalibration baseline uses visible fixed markers and requires no additional arm movements. The tested gain reduction did not defeat the nominal controller; this does not establish that gain adaptation is unnecessary under other conditions. Only four deterministic episodes per condition were used; outcomes are preliminary and do not establish broader reliability or novelty. Nominal kinematics and camera setup are idealized. The full compact per-episode results, traces, model checks and GIFs are in `../results/so101/`, and the frozen cases are in `frozen_comparison.json` and `frozen_transfer.json`.

On the local Ryzen 5 5600X, an initial headless no-render benchmark stepped 2,000 model frames in about 0.025 s; that is not end-to-end throughput. Rendered task episodes took roughly 2–3 seconds wall time each in this run. Simulated time, controller-only computation, total wall time, actions, summed joint movement, clipping, and joint-limit counts are retained per episode. The first matched comparison has four cases per condition and cannot support confidence claims.
