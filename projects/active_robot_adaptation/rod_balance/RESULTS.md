# Rod interaction to idealized balance: first checkpoint

The author Franka rod scene supports physical contact: a diagnostic 20-command push reached the rod at action 9 (zero-based action 8) and displaced it about 11 cm. The former random rollouts did not contact it. In a stationary, zero-action diagnostic, 99.0% of the author's positive finite-difference reward's squared gradient came from one quaternion component. Its cause is not established, and this experiment does not use the reward.

## Task and information boundary

We implemented an independent **idealized support-placement diagnostic**, not robotic pick and place. The author robot executes a fixed observation-guided 20-command push at 10 Hz. It sees the author's noisy 18-dimensional end-effector, joint/gripper and rod-pose features, including simulator-derived pose as a stand-in for perception. It does not receive mass, inertia, COM, contact forces or disturbance labels. A model predicts where along the rod to place a narrow support beneath it. A separate MuJoCo scene resets the rod to rest 0.5 mm above that support and releases it; this teleport/zero-velocity reset supplies idealized transport and orientation. The support location is explicitly distinct from a grasp location. There is no grasp, pick, carry, placement controller, retry or support relocation in the reported trials.

The independent scene uses the author's dimensions and inertial masses: a 0.30 m rod, 0.01 kg rod mass and 0.015 kg fixed internal mass at an offset `theta`. Its total COM is `0.6*theta` along the rod. These values are author-model assumptions, not physical validation. The 24 mm-wide support is centered on the chosen rod location. A trial succeeds only if contact, tilt <0.1 rad, linear speed <0.05 m/s and angular speed <0.5 rad/s hold **continuously for the last 0.5 s of a 2 s release**, sampled at 2 ms. Oracle-COM development checks succeeded for offsets ±0.1 m while centered placement failed at those offsets; these checks verify the basic setup, not a deployable controller. No welds or attachments are used.

The source's blue marker makes its original rendered rod appearance vary with hidden mass. The original controllers never receive pixels. As a separate verification, all 18 held-out push command sequences were replayed with the non-contact marker transparent: every observation and evaluator rod pose matched the original **bit-for-bit**. The marker-hidden GIF is saved with the verification run. This removes the marker from that replay's visible geometry; it does not validate a real camera perception pipeline.

## Hypothesis → intervention → outcome

- **Observation:** the rod moves and rotates under a bounded push, and a narrow support makes the hidden mass distribution consequential. Its response may depend on contact geometry as well as mass.
- **Hypothesis:** observed start geometry and push-induced yaw change can predict a better support location than a fixed prior or current readings alone. Closest foundation: ASID's interaction before identification, but this is our independent estimator and task evaluator.
- **Implementation:** a ridge regression uses initial observed yaw, initial rod-to-tool lateral separation and observed yaw change. The label during offline fitting is the simulator's true COM; that is privileged training supervision, never an evaluation input. The fixed prior is selected by development success across candidate support locations; a zero-centered symmetric prior is also retained. A start-only ridge sees all initial observations; a current-state ridge sees all final observations and estimated linear/yaw velocities. Both use the same regularization, with a feature-scale floor. The current-state and history models receive the **same fixed push data**. No task-specific per-condition recovery rule or deployment weight update is used.
- **Development:** 21 cases, 7 mass offsets and 3 distinct seeds per offset; leave the entire tested offset out of each fitting fold. Center prior 3/21, fitted prior 0/21, start-only 0/21, current-state 2/21, history 9/21, mismatched history 2/21. These are development results, not generalization evidence.
- **Held out:** 6 mass offsets excluded from training, 3 new and distinct seeds each, with one deterministic regression fit. We froze the protocol and code before scoring. Following review, the comparator was expanded **after held-out push collection but before held-out balance scoring**; push contact/displacement diagnostics had been printed. The prior protocol and amendment remain in the project. These are six physical offsets and 18 starts, not 18 training seeds.

| Method | Stable releases / 18 | Mean absolute COM-to-support error | Push actions |
| --- | ---: | ---: | ---: |
| Symmetric centered prior | 0 | 33.0 mm | 0 |
| Development-fitted fixed prior (-35 mm) | 3 | 40.3 mm | 0 |
| Start-only regression | 1 | 48.0 mm | 0 |
| Current-state regression | 4 | 25.3 mm | 20 |
| Interaction-history regression | 7 | 21.5 mm | 20 |
| Mismatched-history negative control | 6 | 30.7 mm | 20 |

Failures are retained in the machine-readable results. Against current state, history alone succeeded on 7 cases, current state alone on 4, and both succeeded on none; against mismatched history, history alone succeeded on 5 and mismatched alone on 4. The shuffled-response control replaces one episode's yaw change with another physical case's yaw change while retaining its initial features; this deliberately creates an out-of-distribution feature combination and is not an ordinary deployable policy. Its 6/18 outcome substantially weakens a claim that the correct interaction history caused the apparent improvement. The sample is small, learned baselines have unequal feature dimension, and the held-out mass distribution gives the fitted empirical prior limited coverage. No general ranking or useful live adaptation is established.

Task-stage simulated time is 2 s placement for the no-push methods and 2 s pushing plus 2 s placement for the interaction methods; all fit within the same 4 s stage budget. These exclude the common author reset/settling (about 4.2 simulated seconds), robot transport, grasp, perception processing and the idealized placement reset, so they are **not** full manipulation times. All 20 push commands are counted; no retry or recovery occurred. The 21 development push episodes took 33.0 s host wall including environment setup; 18 held-out pushes took 27.85 s; the 108 release evaluations and fitting took 4.28 s wall including two rendered recordings. Every balance warning counter was zero. Push contact was sampled at action boundaries, not every physics step; joint-limit and prohibited-self-collision physics-step audits were **not** performed, so their counts are unknown. The release has no robot joints. These limitations preclude safety claims.

## Verification and revision record

An independent reviewer inspected the first development result. It found that the 14/14 result with two repeated noise/start seeds was confounded: a start-only model also achieved 14/14 by exploiting an ill-conditioned fit and correlated reset differences. That result is invalid as evidence of interaction value. The revised data use distinct seeds for every mass/start case; training and test mass offsets and seeds are asserted disjoint for held-out scoring. Ridge standardization has a floor, and the start-only, richer current-state, centered-prior and mismatched-history controls were added. The reviewer also prompted the half-second dwell. The earlier runs remain on disk, and the relevant data and record are preserved.

The independent review covered development and the code; it was not a complete independent audit of the final held-out run. A separate local verification checked 18 visual-marker-hidden replays against all stored observations and truth poses, source hashes against the frozen configuration, disjoint seeds/offsets, no simulator warnings in release, and syntax. Numerical timestep sensitivity, physics-step collision/limit auditing, and the source's derived-model-constant handling remain open. Exact correspondence of observation noise between a simulator stand-in and real perception is unvalidated.

## Decision and next test

**Inconclusive:** the interaction model selected different support positions and achieved 7/18 releases, but the mismatched-response control reached 6/18 and privileged training labels are required. Do not promote this estimator as a shared adaptation mechanism. The next focused experiment should use paired, explicitly randomized histories on a larger set of independent starts and evaluate a simple calibrated physical-response estimator against the same push histories, measuring support success and the cost of the push. Require correct-history versus mismatched-history paired benefit before transferring the estimator; keep the idealized reset and simulator limitations explicit.

## Reproduction

Set up pinned upstream source and isolated environment using [the existing setup](../published_manipulation/README.md). From the project root:

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl PYTHONPATH=/tmp/active-asid-src /tmp/active-asid39/bin/python rod_balance/diagnose.py
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl PYTHONPATH=/tmp/active-asid-src /tmp/active-asid39/bin/python rod_balance/collect.py --set development
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl PYTHONPATH=/tmp/active-asid-src /tmp/active-asid39/bin/python rod_balance/collect.py --set evaluation
```

The commands create new uniquely named runs; pass their `summary.json` paths in place of the recorded paths to score a rerun:

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 /tmp/active-asid39/bin/python rod_balance/compare.py --mode evaluation --train results/published_runs/20261117T183441.744535Z_91b38b9b1287_rod_push_development_aa07c4fe/summary.json --data results/published_runs/20261117T183600.677617Z_91b38b9b1287_rod_push_evaluation_3024b01b/summary.json
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl PYTHONPATH=/tmp/active-asid-src /tmp/active-asid39/bin/python rod_balance/verify_replay.py --data results/published_runs/20261117T183600.677617Z_91b38b9b1287_rod_push_evaluation_3024b01b/summary.json
```

Outputs: [diagnostic trace](../results/published_runs/20261117T183011.442401Z_91b38b9b1287_rod_contact_diagnostic_939d3520/trace.json), [revised development](../results/published_runs/20261117T183709.473047Z_91b38b9b1287_rod_balance_development_d62f7429/summary.json), [held-out comparison](../results/published_runs/20261117T183724.103754Z_91b38b9b1287_rod_balance_evaluation_7636f12a/summary.json), [marker-hidden verification](../results/published_runs/20261117T183810.100400Z_91b38b9b1287_rod_marker_hidden_replay_09dba646/summary.json), and [frozen protocol](FROZEN_EVALUATION.json).
