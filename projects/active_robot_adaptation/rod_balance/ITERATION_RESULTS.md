# Rod information-to-placement iteration

**Decision: attribution remains inconclusive.** In the frozen 48-case comparison, the history model lowered mean absolute support error relative to final-state feedback (paired difference -9.86 mm, bootstrap 95% interval -15.58 to -4.34 mm) and produced 20 rather than 16 stable releases. On the same rods, however, it gained 10 releases and lost 6; the paired success-difference bootstrap interval includes zero. More importantly, a post-hoc correction for omitted finite-difference joint velocities gave 17 stable releases for each model, with a paired error interval also spanning zero. The correction was made after seeing frozen outcomes and is exploratory, not a replacement frozen result. We cannot yet say which interaction information improves stable release reliably.

## Evidence carried forward

The [old paired table](SAVED_PAIRS.md) lists all 18 prior evaluated rods, including the seven historical response-only gains and four losses against final-state feedback. Those cases and the earlier 21 training cases are **development data** here. In the old interface, current state was the final 18-element author observation (end-effector position, robot joints/gripper, rod pose) plus end-effector and rod finite-difference linear velocities and rod yaw velocity. It omitted finite-difference joint velocities. The prior historical method used starting yaw, final-minus-starting yaw and initial rod/tool lateral separation; in this iteration that is method D, the simple response estimator.

The new C model receives B's final observation and velocities, starting yaw and rod/tool lateral separation, four intermediate yaw responses, total commanded x/y displacement and observed net rod x/y displacement. Commands are displacements, **not measured forces**. Both regressions and D use the same 39 development episodes and same regularization; true COM labels are privileged training supervision and never evaluation inputs. D does not use the full time series. A was selected by leave-one-mass-offset-out development results: start-only 5/39, centered prior 3/39, fitted prior 0/39. The simple empirical fitted prior is weak on the sparse development mass set; its 0 count does not imply that priors generally cannot work.

## Frozen evaluation

The [frozen protocol](ITERATION_PROTOCOL.json) generated 48 continuous hidden mass offsets uniformly in ±0.1 m and independently generated 48 pose/noise seeds, with separate RNG streams. They were fixed before the evaluation results were inspected. Each physical push was collected once and used by all push-based methods. A executes no push. Each release uses exactly the previous evaluator: a rod placed at rest 0.5 mm over a 24 mm-wide support, simulated for 2 s; support contact and declared tilt/velocity bounds must hold for the last 0.5 s. This is a **simulated robot push followed by an idealized release**, not a pick, carry or physical trial.

| Method | Stable / 48 | Mean absolute support error | Push actions | Stage simulation time |
|---|---:|---:|---:|---:|
| A: immediate start-only | 8 | 35.86 mm | 0 | 2 s |
| B: pushed, final state | 16 | 31.89 mm | 20 | 4 s |
| C: pushed, state plus history | 20 | 22.03 mm | 20 | 4 s |
| D: pushed, calibrated response | 13 | 20.02 mm | 20 | 4 s |

The above 2/4 s include release and, where applicable, 2 s of push; they exclude the common reset (~4.2 s simulated), perception overhead and the unmodeled robot transport/placement. There were no retries. Wall time to collect the 48 instrumented pushes was 78.97 s including environment setup; see the score manifest for comparison compute. All 48 contacted the rod at some point. Physically simulated joint velocities in the evaluator are **not** controller inputs; the velocity estimates come from noisy position observations.

Paired bootstrap intervals resample the 48 independently sampled rods 10,000 times. Error difference is first method minus second: negative favors the first. Additional paired success intervals below were derived post hoc; the original frozen score and its predeclared bootstrap are preserved.

| Pair | Mean absolute-error difference, mm (95% interval) | Stable-release gains / losses |
|---|---:|---:|
| B − A | -3.97 (-13.06, +4.92) | 12 / 4 |
| C − A | -13.83 (-21.18, -6.47) | 15 / 3 |
| D − A | -15.84 (-22.01, -9.58) | 9 / 4 |
| C − B | -9.86 (-15.55, -4.49) | 10 / 6 |
| D − B | -11.87 (-19.89, -4.42) | 6 / 9 |
| C − D | +2.01 (-2.97, +7.26) | 11 / 4 |

For C − B, the post-hoc 95% bootstrap interval for the success-rate difference is -8.3 to +25.0 percentage points; it is not strong evidence of a release advantage. D's lower mean error does not mean more stable releases. The support is only ±12 mm from its center: C placed 20 cases within 12 mm of evaluator COM and D placed 15. D reduced some larger errors without moving as many cases across the support threshold; additionally 2 D placements within 12 mm failed the dynamic dwell, as did 1 B placement. This is a finite-sample diagnostic, not a claim that ±12 mm guarantees stability.

The [fresh per-rod paired table](ITERATION_PAIRS.md) retains every success and failure and marks each B-to-C gain or loss. For fresh case 6, seed 259722, observed yaw changed from 0.762 to 0.452 rad (-0.310 rad); initial rod/tool lateral separation was 17.8 mm. B selected +6.6 mm and failed, C selected -47.7 mm and succeeded, and D selected -60.0 mm and succeeded. The evaluator-only COM was -53.2 mm. The same 20-action push generated all three observations. This is an illustrative case, not a general explanation of the score.

## Attribution and verification

Ten diagnostic donor draws per recipient reassign intact recorded action/response sequences from eight nearest starts in observed initial geometry while keeping the recipient's own initial and final observations. Correct C pairing produced 20 successes; the ten reassigned aggregate draws produced 8–13, mean 10.7. Correct pairing had 22.03 mm mean error versus 35.86 mm averaged over resampling. Donors are independently sampled, not a bijective permutation. The median mismatch between the donor's yaw response and the recipient's own initial/final yaw was 0.177 rad. Such inputs can be inconsistent with the preserved current observation, so this is an out-of-distribution diagnostic and does not isolate information causally.

The frozen B and C omitted velocities derivable from the last two robot joint observations. The [post-hoc velocity check](../results/published_runs/20261122T212517.304304Z_aa04a08d803e_rod_velocity_posthoc_5f4741a5/summary.json) adds the same eight joint finite differences to both: B and C each scored 17/48; C alone succeeded 8 and B alone 8. Mean errors were 32.42 and 27.19 mm; C − B was -5.22 mm (95% bootstrap interval -11.41 to +0.96 mm). This is exploratory because frozen results had already been inspected, and no new independent evaluation was performed.

The audit split each push's MuJoCo stepping into individual physics steps. A two-case replay of old pushes was bit-for-bit identical in all observations to the original batched stepping. The 48 fresh pushes contained 48,000 audited physics steps: **5,750 had at least one joint-limit excursion >0.00001**, all in finger slide joints; maxima were 0.979 mm and 0.937 mm for fingers 2 and 1. No arm-joint excursions over that threshold and no prohibited nonadjacent self-collision steps were observed. Same-body, parent/child and left/right finger-pair contacts are excluded; robot/rod contact is counted separately. The author's reset/settling is outside this audit. No MuJoCo warning counters were raised. These are simulated soft-limit excursions, not a physical safety certification.

Marker visualization was hidden during the fresh pushes; observations remain author simulator-derived noisy features, not a validated image-perception pipeline. No independent reviewer audited this iteration. Frozen source hashes, 48 distinct seeds and physical offsets, paired case counts, limit audit and numerical warnings were checked locally. The source-model inertial constants and simulation-to-hardware validity remain limitations.

## Next falsifiable test

The next observation worth testing is **yaw measurement noise**, without an additional physical movement. Against evaluator-only truth, the observed start-to-end yaw change had 0.0357 rad mean absolute error and 0.0679 rad 90th-percentile error; mean absolute error was 0.0298 rad on C successes and 0.0399 rad on C failures. These are post-hoc associations, not proof of cause. On a new independent set, use the same fixed push and estimate yaw from a short window of observed poses, fitting all models on development only. Prediction: windowing lowers yaw-response error by at least 20% and gains at least four net stable releases over the unfiltered matched estimator, without extra actions; failure of either prediction argues against this mechanism. The evaluator may use truth for this diagnostic only. If the controlled test fails, close this rod branch rather than adding a larger model or another disturbance sweep.

## Reproduction

From `projects/active_robot_adaptation`, with the pinned ASID setup in [the earlier instructions](../published_manipulation/README.md):

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 /tmp/active-asid39/bin/python rod_balance/iteration.py dev
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl PYTHONPATH=/tmp/active-asid-src /tmp/active-asid39/bin/python rod_balance/iteration.py collect --n 48 --protocol rod_balance/ITERATION_PROTOCOL.json
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 /tmp/active-asid39/bin/python rod_balance/iteration.py score --data results/published_runs/20261122T212216.364638Z_aa04a08d803e_rod_iteration_push_d192d75c/summary.json --protocol rod_balance/ITERATION_PROTOCOL.json
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 /tmp/active-asid39/bin/python rod_balance/velocity_check.py --data results/published_runs/20261122T212216.364638Z_aa04a08d803e_rod_iteration_push_d192d75c/summary.json
```

Each command creates a unique run directory; use the newly collected `summary.json` in a new score run. The [raw 48 push traces](../results/published_runs/20261122T212216.364638Z_aa04a08d803e_rod_iteration_push_d192d75c/summary.json), [frozen scores and donor assignments](../results/published_runs/20261122T212341.239532Z_aa04a08d803e_rod_iteration_scored_5b9ef5c7/summary.json), [post-hoc paired statistics](ITERATION_ANALYSIS.json), source snapshots, protocol, and a marker-hidden push GIF are retained.
