# Shared manipulation pilot results

## Decision

A shared fixed-window predictor was trained and compared in closed loop, but this experiment does **not** show useful adaptation from history. All four methods succeeded on every tested guided push and pull, and the learned methods chose the same action sequences as conventional feedback in all 360 paired task-step comparisons (within 1e-7 m). History reduced one-step prediction error for two of three training seeds and increased it for one; that did not change action selection or success. Do not add probing or a disagreement policy on this evidence.

The next hypothesis is narrower: ordinary feedback may remain sufficient because these guided tasks need only one or three nonzero task actions and provide nearly no opportunity for useful context. First verify action sensitivity and collect diverse contact transitions within the existing push/pull tasks, including changes after context initialization, under a frozen protocol. Address free-push numerical stability and the known bottle enclosure problem within their existing task families before claiming cross-task transfer. No held-out task-family, hardware, physical generalization, or novelty claim is made.

## Task and information boundary

Both tasks use the existing pinned MuJoCo SO-101 and joint-position actuator limits. Following two free-box numerical-instability attempts, pushing was **simplified to a guided sliding object**; pulling uses a separate guided slider with a physical handle, grasped by the jaws. This substantially narrows the distinction between tasks and does not validate free-object pushing. The bottle/lift task was deferred because of the existing closure displacement, unresolved empty/wet feasibility and joint-limit excursion issues; no fabricated successful third task or task-family holdout is reported.

The controller has 320×240 overhead RGB features, calibration from the same fixed table fiducials, six joint positions, nominal FK tip xy, previous commands and timestamps. Perception supplies object xy, finite-difference visual velocity, and a visibility bit; missing visual xy/velocity are zero with mask zero. No hidden mass, friction, resistance, contact force, actual object pose or disturbance label is given to a controller or predictor. The pull slider carries a rigid cyan fiducial 45 mm from its handle to remain visible; its known nominal offset is used by the shared perception interface. Push uses the existing green detector. These are task-specific perception/setup scaffolds shared by all methods. If the feature is missing the action selector waits, rather than substituting truth. No missing visual observations occurred in the final data or evaluation.

Actions are x/y end-effector displacement, limited to ±12 mm per axis at 0.1 s intervals, with existing 15 mm vector clipping, fixed wrist roll/approximate pitch, and a fixed task-setup jaw position. Neither arbitrary 6-DoF control nor force control is assumed. The supplied task-specific setup moves the arm and, for pulling, closes the jaw; these actions, the initial 0.5 s settling, task motion, and final 0.5 s settling are included in recorded budgets and time. All methods get the same setup and goal: move the observed object 40 mm in negative world y from its initial observed position. Evaluator success is final actual y within 12 mm of the corresponding true initial-position-relative target. There is no within-episode change, so *recovery time is not measured*; recorded simulated duration is completion/budget time.

A separate physics-step development check recorded actual gripper–object contact on push and contact with both fixed and moving gripper bodies on pull; after three task actions each slider moved about 31 mm. This check used simulator truth only for logging and had no numerical, self-collision or joint-limit warnings. There is no attachment, weld, oracle control or external force.

## Dataset and training

`shared_protocol.json` was frozen after correcting MuJoCo derived mass constants with `mj_setConst` and discarding prior inspected test cases as development. The corrected dataset contains **24 independent episodes**: 18 train, two validation and four held-out test episodes, each with ten transitions. Train/validation/test use disjoint physical-parameter tuples and initial positions. One test condition (friction 0.75, mass scale 1.25, slider resistance 0.025) and two starts per task are too few for a population estimate. All 24 episodes completed setup, with no flagged numerical instability, missing feature, prohibited self-collision or joint-limit excursion physics steps. Earlier invalid free-object attempts and the partially inspected pre-correction evaluation are preserved separately.

Trainable targets are differences between consecutive **image-derived** feature positions; no simulator-truth training target is used. The model fits 180 valid training, 20 validation and 40 test transitions. Fixed hand-chosen unit scales are used; normalization is not fitted to validation/test. Samples are correlated within episodes, and the 40 test transitions come from only four episodes. Dataset `episodes.json` SHA-256: `939a5b588112ed803908aea92e0eb5928a502511b5b8c00219d9fefe81ec505f`.

The current-only predictor takes 13 current features, including visual velocity, and two action values. The history model additionally receives three previous observation/action vectors; both use the same 32-unit tanh MLP form, with 578 and 2,018 weights respectively. Thus architecture is matched, though input dimensionality and parameter count differ. No weights change during evaluation; history updates from observations. The frozen-history ablation uses the same trained history weights but freezes context after three actions, introducing a distribution shift. Actions are selected from five bounded y candidates at horizon one by predicted next observable position and the shared goal cost. No gradients are taken through contact.

The predeclared training used 400 NumPy Adam updates per model, batch 64 and seeds 1/2/3. Finite-difference checks passed and a separate eight-sample overfit smoke run reduced loss. Tinygrad's sandbox GPU discovery segfaulted; CPU mode lacked clang, so the pilot used NumPy rather than expanding infrastructure work. Existing Tinygrad work is unchanged. For this small local CPU workload, each 400-update training run took 0.036–0.041 s (about 0.63–0.72 million sampled examples/s); peak process RSS was about 37 MiB. This is not a GPU benchmark. Saved model checkpoints are a few kilobytes each.

| Seed | Current-only test RMSE, mm/coordinate | History test RMSE, mm/coordinate |
|---:|---:|---:|
| 1 | 4.61 | 4.51 |
| 2 | 4.15 | 4.34 |
| 3 | 4.42 | 4.26 |

These are one-step image-feature prediction errors; better prediction alone is not task recovery. Validation/test errors and losses for all seeds are in the machine-readable training summary. Only one held-out physical tuple was tested.

## Matched closed-loop comparison

There are two fixed starts per task, one held-out physical tuple, three model seeds, and four methods: **48 episodes**. Seeds change trained weights; the repeated conventional episodes are deterministic repetitions of the same two starts, not six independent physical cases. All methods share action and movement limits, calibration, perception, goal, memory-reset rules and setup. There are no retries.

| Task | Conventional | Current-only | History | Frozen history |
|---|---:|---:|---:|---:|
| Guided push | 6/6 | 6/6 | 6/6 | 6/6 |
| Handled-slider pull | 6/6 | 6/6 | 6/6 | 6/6 |

All pairs ended at the same recorded positions: median error was 11.08 mm for push and 8.01 mm for pull, with push near the 12 mm success boundary. Median total actions were 25.5 and 42.5, movement 1.363 and 1.305 rad, and simulated time 3.05 and 4.75 s respectively, including setup and settling. Task-action sequences consisted of three nonzero commands for each push and one for each pull, followed by zero commands once within observed tolerance. Because useful movement ended before the three-action context freeze could affect decisions, the frozen-history ablation is largely uninformative. All 48 episodes had zero logged numerical invalidity, prohibited self-collision physics steps, or joint-limit violation physics steps.

The five-candidate, one-step selector took a median 0.055–0.064 ms per counted task step for learned push controllers and 0.021–0.026 ms for learned pull controllers, averaged over all ten task steps, including zero-action steps; it is not isolated per-inference latency. Controller-only model selection excludes simulation and rendering. Whole-episode wall time medians were approximately 3.6–4.3 s on the local CPU and are affected by rendering and shared-machine load. Full timings, actions, observations, limits, episodes, model weights, source snapshots, versions and seeds are saved.

## Artifacts and commands

Dataset: `results/so101_runs/20261116T215911.076327Z_aa753c5c56e8_shared_data_54a25f75/`.
Training: `results/so101_runs/20261116T220051.143395Z_aa753c5c56e8_shared_train_58a691af/`.
Evaluation: `results/so101_runs/20261116T220056.557593Z_aa753c5c56e8_shared_evaluation_39707661/`.
Analysis and prediction chart: `results/so101_runs/20261116T220516.269457Z_aa753c5c56e8_shared_analysis_13d3fa74/`.
Contact check: `results/so101_runs/20261116T220204.386910Z_aa753c5c56e8_shared_contact_check_991fea93/`.
Each includes revision, source snapshot, configuration and seed metadata. Eight short evaluation GIFs show both tasks and four methods. The earlier failed analysis was preserved; the corrected analysis is above.

From the project directory, after installing the existing SO-101 environment and fetching the pinned model as described in `README.md`:

```bash
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/shared_contact_check.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/shared_collect.py
OPENBLAS_NUM_THREADS=1 PYTHONPATH=so101 .venv/bin/python so101/shared_train.py --data <new-data-run> --updates 400 --seeds 1 2 3
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/shared_evaluate.py --models <new-training-run> --seeds 1 2 3
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/shared_summary.py --evaluation <new-evaluation-run> --training <new-training-run>
.venv-so101/bin/python so101/shared_plot.py <new-analysis-run>
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python -m unittest discover -s so101 -p 'test*.py'
```

Each run creates a unique directory and snapshots uncommitted source. A previously interrupted free-object data run and a pre-correction comparison are preserved as development evidence and excluded from the final models. No paid compute, purchases, physical operation, RL training, or Git push was used.
