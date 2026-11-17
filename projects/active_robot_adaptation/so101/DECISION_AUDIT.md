# Decision sensitivity and slider ambiguity

## Decision

**No meaningful adaptation opportunity emerged in this bounded guided-slider investigation.** Close this guided-task branch for the tested action set, horizon, objective, and two resistance values. The matched slider states and their recent action/observation histories produced the same best one-step action under both dynamics. The largest simulator-truth difference between corresponding candidate displacements was 0.243 mm; the largest visual-feature difference was 0.189 mm. These are outcomes for three states and five actions, not bounds for other situations. The camera scale is approximately 2.3 mm/pixel at the marker plane; a centroid can have subpixel precision, so the small difference is not a proof of theoretical unobservability.

The existing bottle-grasp interaction is a more concrete next candidate, without introducing a new task family: in the prior recorded empty-bottle `holdout-a` pair, the same nominal closure displaced the dry bottle 44.1 mm and the wet bottle 77.5 mm; dry was acquired and wet escaped. The decision question is whether to continue closure or stop/reposition from permitted visual and jaw observations. That is a **hypothesis**, not a demonstrated optimal-action difference: empty/wet feasibility is unresolved, the rigid model has no crushing cost, and the earlier more-closed/slower rule did not solve it. First establish an attainable enclosure and evaluate that simple feedback decision; do not assume a new history model or probe is necessary. See `BOTTLE_RESULTS.md` for the source evidence.

## Two focused sessions

1. **Recorded-decision audit.** Replayed the previously evaluated seed-1 history-controller episode at the first held-out start for each of push and pull. Inspected decision indices 0, 1 and 3, including states where the goal gate suppresses planning. These cases are development evidence, not a fresh performance evaluation. The original episode, model checkpoint, revision, seed, and source snapshot are identified in `decision_audit_protocol.json` and run metadata.
2. **Slider ambiguity.** Tested the frozen resistance pair 0.01 N / 0.04 N with fixed mass scale 1.25 and friction 0.75, using the same initial start, goal and two preceding -6 mm actions. Checked both independently executed natural trajectories and a counterfactual in which *the full same physical state* is evaluated under each resistance model. Neither experiment established different practically consequential best actions. Per the predeclared criterion, no sequential-target task, new training, online estimator, or after-change recovery comparison was started. This avoids using a dynamics change to create an unsubstantiated adaptation opportunity.

## What actually selects an action

The controller acts every 0.1 s. Its five world-y candidate displacements are -12, -6, 0, +6 and +12 mm, with x fixed at zero. Each candidate is clipped componentwise to ±12 mm, then the existing Cartesian controller applies a 15 mm vector cap and joint increments are clipped to ±0.08 rad. The nominal task setup is scripted and task-specific, including closing the gripper for pull; all methods receive the same setup and current image-derived features.

When the feature is visible and the estimated y error is at least 12 mm, the learned prediction is actually used: candidate score is one-step absolute predicted visual y error plus `0.001 * abs(action_y)` in metres; predicted y displacement is clipped to ±30 mm. `argmin` selects the first candidate on an exact tie. The horizon is one action, with no prediction of later settling. If the feature is missing or estimated error is below 12 mm, the code instead commands zero **without computing candidate predictions**. The gate is part of the existing task/controller, and action scores shown for gated decisions are explicitly hypothetical. The earlier pilot's pull needed one nonzero task action, push three; all four methods succeeded under that 12 mm success criterion.

For each audited state, comparisons used exactly the same candidate set, horizon, goal and score: the trained seed-1 current-only and history predictors; a nominal 1:1 response predictor (predicted object y displacement equals requested y displacement); and evaluator-only one-step simulator rollouts. The latter are diagnostic lookahead, not an eligible deployment policy and not proof of global optimality. Actual actions and future camera observations are used only for retrospective scoring. No controller receives physical parameters, contact forces or true slider poses.

| Recorded state | Goal gate bypasses planning | History vs rollout ranking inversions /10 | History choice | Rollout choice | One-step score difference for history choice |
|---|---|---:|---:|---:|---:|
| Push, step 0 | No | 0 | -12 mm | -12 mm | 0 mm |
| Push, step 1 | No | 4 | -12 mm | 0 mm | 0.012 mm |
| Push, step 3 | Yes | 2 | -12 mm | -6 mm | 4.17 mm, hypothetical |
| Pull, step 0 | No | 0 | -12 mm | -12 mm | 0 mm |
| Pull, step 1 | Yes | 3 | 0 mm | -6 mm | 5.78 mm, hypothetical |
| Pull, step 3 | Yes | 0 | -6 mm | -6 mm | 0 mm, hypothetical |

Ranking inversions compare the ten candidate pairs; even subpixel score differences count. At push step 1, all five simulator candidates moved the object the **same** -14.086 mm in the next 0.1 s despite different commands; the tiny preference for zero comes entirely from the action penalty. The history model's one-step displacement RMSE across those five counterfactual actions was 5.34 mm, but its selected action incurred only 0.012 mm extra score. None of the 90 diagnostic candidates saturated the joint increment bound.

At push step 3 the recorded action was zero because the observed goal error was already inside the gate. If the gate were removed just for an offline one-step calculation, a -6 mm candidate scored 4.72 mm versus 8.74 mm for zero. Both are within the current 12 mm task tolerance; no longer-horizon or settled outcome was tested. This shows that predictions *can* matter for a different precision objective, but does not establish improvement in the existing success metric or justify silently changing it.

## Full-state isolation and physical evidence

Each candidate rollout uses a separate environment/model with a separate `MjData`. Before every candidate, `mj_copyData` copies **all** MuJoCo data, including simulation time, position, velocity, actuator state, controls, applied forces, solver and warmstart state. The perception history and current observation are copied separately. Source integration-state hashes and source observations/counters were checked unchanged after all five candidates. For every replayed decision, advancing the source with its recorded action exactly matched the corresponding cloned rollout's next 13 visual/joint features (maximum absolute difference 0); original recorded observations also replayed exactly. This checks isolation and replay, not real-world dynamics.

Both natural paired setups converged, with fixed and moving gripper bodies in actual contact with the handle. After setup and two -6 mm actions, both stayed within the slider's ±100 mm range; no numerical warnings, prohibited self-collision physics steps, or joint-limit violations were recorded. Natural pair visual position and visual velocity features were identical at the three sampled states; the maximum nominal FK tip difference was 0.053 mm and the maximum joint-reading difference was 0.000524 rad. The small encoder/FK differences mean the observations were not exactly identical, and noise was not modeled. Earlier action/observation transitions were also visually identical at these samples, so they did not demonstrate usable distinguishing information.

In the exact-state counterfactual, copied integration states were bit-identical before testing both resistance models; only the hidden resistance differed. The shared prehistory was generated under low resistance, so the high-resistance branch is a counterfactual rather than an independently reached high-resistance trajectory; the natural pair above supplies that separate check. Across the five candidates at each of the three states, simulator rollout selected -12, -12 and -6 mm respectively in **both** conditions. The largest true displacement difference was 0.243 mm and the largest difference in extracted visual feature was 0.189 mm. The two resistance values are a limited modeled comparison, not measured hardware properties. Joint friction loss is a slider force parameter, not a label supplied to a policy. It may be overwhelmed here by the grasp and position servo; that is a plausible interpretation, not a separately identified physical cause.

All 18 audited states × five candidates completed without numerical warnings, prohibited self-collisions or joint-limit violations at physics-step resolution. Mean time to score both learned predictors plus nominal for five candidates was 0.431 ms; five cloned simulator rollouts, including physics and rendering, averaged 728 ms. The whole diagnostic took 33.24 s before video encoding on the local CPU. Diagnostic lookahead time is not charged as a deployment controller or used to score task success.

## Reproduction and evidence

The final immutable-by-convention audit is `results/so101_runs/20261117T003539.496461Z_fd2c76dfeddd_decision_audit_8e14a5b1/`; it includes full candidate predictions, costs, physical effects, safety counters, state checks, timings, source snapshot and a recording of the six replayed decision states. `results/so101_runs/20261117T003650.515033Z_fd2c76dfeddd_decision_summary_72871be1/` contains compact derived results and `decision_curves.svg`. `results/so101_runs/20261117T003750.805800Z_fd2c76dfeddd_decision_feasibility_5485fc24/` records the independently rerun paired setup, actual jaw contacts and physical limits. Earlier audits remain separately preserved; reruns verified state copying and added timing/saturation instrumentation, without changing the frozen conditions or policy. No new evaluation set or training data were generated.

From the project directory:

```bash
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/decision_audit.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/decision_feasibility.py
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python so101/decision_summary.py results/so101_runs/20261117T003539.496461Z_fd2c76dfeddd_decision_audit_8e14a5b1
.venv-so101/bin/python so101/decision_plot.py results/so101_runs/20261117T003539.496461Z_fd2c76dfeddd_decision_audit_8e14a5b1 <new-summary-run>
MUJOCO_GL=egl PYTHONPATH=so101 OPENBLAS_NUM_THREADS=1 .venv-so101/bin/python -m unittest discover -s so101 -p 'test*.py'
```

Every run creates a unique directory; plotting requires a new target. This diagnostic does not establish that history adaptation is unnecessary on other tasks, parameters, models or hardware.
