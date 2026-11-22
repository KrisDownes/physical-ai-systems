# Saved rod review

## Corrected comparison

This is an exploratory review of the 48 rods in the frozen [iteration protocol](ITERATION_PROTOCOL.json). The original push dataset and frozen scores remain at their original paths. The [velocity correction](velocity_check.py) followed inspection of those scores. Its saved predictions and all 96 release dwell counts were reproduced using the existing evaluator.

- Both models use the same final 18 observations: tool XYZ, seven arm joint positions, the gripper field, and rod XYZ plus a quaternion. The author wrapper adds Gaussian noise to the flattened features with standard deviation 0.01 in their native units. In this 2-D configuration the source sets `gripper: false` and appends the last element of the pose as the gripper field. That value is an orientation component, with fixed commanded orientation and noisy observed motion. The rod pose is a noisy simulation proxy for perception. Accordingly, the eighth value in the joint finite difference is a derivative of that field, not a measured jaw velocity.
- Both models use samples 19 and 20 at 1.9 and 2.0 seconds relative to the first command. Sample 0 is immediately after reset. The timestamps are inferred from the 10 Hz control period and verified physics-step schedule. The models use backward differences divided by 0.1 seconds for tool XYZ, rod XYZ, seven arm joints and the extra gripper field. The yaw calculation normalizes each quaternion and wraps the yaw difference before dividing by 0.1. The raw final quaternion remains in both input vectors. No sample after the placement decision is available.
- C adds initial yaw, initial rod-to-tool lateral offset, four yaw changes relative to the initial sample at steps 5, 10, 15 and 20, summed XY displacement commands, and the observed net XY displacement of the rod. Commands are position increments. The shared 33 input values were checked for exact equality on each case. Changing evaluator-only fields left both feature vectors identical.
- The fits use the same 39 development histories and COM labels from earlier runs, with disjoint masses and pose/noise seeds from the 48 scored rods. The fit centers and scales each feature using only development statistics, with a 0.02 scale floor and ridge penalty 1 on the feature coefficients. Both clip the predicted support point to ±60 mm. The physical release initializes a resting rod 0.5 mm over the 24 mm-wide support and evaluates support contact, tilt and speed continuously over the final 0.5 seconds of a 2-second release. It uses the simulated mass in physics after the decision. There are no future samples, contact forces, or COM labels in deployment features.

| Corrected method | Stable / 48 | Mean absolute support error | Median absolute support error |
|---|---:|---:|---:|
| B · final state and velocities | 17 | 32.42 mm | 17.61 mm |
| C · same state and velocities plus history | 17 | 27.19 mm | 15.97 mm |

Paired C minus B absolute error was **−5.22 mm mean** (95% interval −11.41 to +0.96) and **−3.41 mm median** (95% interval −14.33 to 0.00). The median is calculated on case differences. Intervals use 10,000 paired bootstrap draws resampling complete rods. C gained 8 and lost 8 stable releases, with 9 shared successes and 23 shared failures. The saved success-difference interval is −16.7 to +16.7 percentage points.

- Gained by C: 3, 6, 9, 13, 30, 34, 38, 41.
- Lost by C: 1, 11, 12, 19, 24, 37, 40, 44.
- Stable with both: 0, 4, 21, 25, 26, 29, 36, 39, 46.

Case 6 remains a history-specific win **versus B**. B chose −16.28 mm and failed, C chose −60.00 mm and held for all 250 dwell steps, and the COM was −53.21 mm. D, the simpler yaw-response fit, also chose −60.00 mm and held. Observed yaw change was −0.310 rad against simulated −0.281 rad. Exact yaw still leaves B failing and C and D stable.

## Yaw attribution

The observed yaw-change error averaged 0.024 rad on the eight C-only wins and 0.052 rad on the eight losses. This descriptive difference can be driven by cases such as case 1, which had −0.297 rad observed and −0.127 rad simulated. The per-case JSON retains both values, the choices and release outcome for all 48 cases.

The diagnostic holds the fit, commands, initial state and other observed quantities fixed. One intervention substitutes exact relative yaw response in the history features of C or the simple response D. The other replaces all yaw components while preserving the observed tilt, quaternion norm and other features. It then recomputes features and reruns the same release. The exact-yaw substitutions use evaluator truth only here. They are sensitivity tests with the existing fitted models, whose inputs during fitting were noisy. The results give no general upper bound on a refitted noise filter.

| Diagnostic substitution | Rescued cases | Newly failed cases | Net stable releases |
|---|---|---|---:|
| C · exact response only | 1, 31 | 26 | +1 |
| C · all yaw exact | 31 | 13, 26, 41, 46 | −3 |
| D · exact response only | 1, 31, 37 | 16, 27, 46 | 0 |
| B · all yaw exact | none | 0, 4, 24, 25, 26, 29, 37, 40, 46 | −9 |

Case 31 shows a real measurement-sensitive miss. C observed +0.083 rad of rotation versus +0.161 rad in the simulator. Its original placement missed COM by 11.75 mm, inside the support's nominal 12 mm half-width, yet it settled tilted 1.084 rad with the rod still touching the support. Exact response placed it 9.39 mm from COM, with all 250 steps stable and final tilt 0.007 rad. For a near-edge success, B on case 29 held at 11.33 mm error with final tilt 0.008 rad. This is B's largest-error success and exceeds many successful placements, although it is slightly closer than the low-error failure. The corrected B/C outcomes contain no reversal where a successful absolute error exceeds a failed one. In this pair the successful placement begins 0.67 mm inside the nominal edge, versus 0.25 mm for the failure. Contact by itself does not meet the stability criterion.

## Finger actuation

Replaying the 48 saved command sequences reproduced every recorded observation and evaluator pose exactly. It also reproduced the per-case limit and contact counters. The new instrumentation records 2 ms intervals within each push. It associates post-integration joint positions with the contact set from the same integration step, so alignment is at physics-step resolution. The reset is outside these measurements.

The 5,750 limit steps above the 0.01 mm audit threshold sum to 11.500 seconds across the 48 two-second pushes. Every rod had at least one excursion. Median duration was 0.262 seconds per rod, and the longest continuous interval lasted 0.612 seconds (case 40, 1.328 to 1.940 seconds). The excursions were entirely below the zero-travel end of the finger joints. Maximum depth was 0.937 mm for finger 1 and 0.978 mm for finger 2. The latter occurred during finger–rod contact. The original audit recorded zero prohibited nonadjacent self-collision steps.

| Timing of above-threshold limit steps | Steps | Duration across pushes |
|---|---:|---:|
| Before first robot–rod contact | 593 | 1.186 s |
| Simultaneous with robot–rod contact | 4,696 | 9.392 s |
| Out of contact between first and last contact | 395 | 0.790 s |
| After last robot–rod contact | 66 | 0.132 s |

All 4,696 simultaneous excursion steps included finger–rod contact, across 43 cases. The median first contact was at 0.879 seconds. These timings show that closing-joint behavior overlaps the measured interaction.

The author actuator sends its closed target to the end of a soft constraint. In a diagnostic counterfactual, moving only the finger actuator's nominal target **1 mm inward** with all 20 recorded Cartesian commands fixed changed final rod position by 1.47–6.80 mm across case 6 and the five rods selected for largest recorded excursions. It changed signed yaw response by up to 0.042 rad. In case 6 that moved exact rotation from −0.281 to −0.239 rad, while contact duration increased from 0.782 to 1.048 seconds. This sensitivity is on the scale of measured yaw error. The actuator change also changes contact geometry and compliance, so the diagnostic cannot assign the change solely to penetration past the stop. Its 1 mm target reduced excursions on five of six cases, but made case 40 worse (337 to 373 steps), so it is not a sufficient fix.

If this interface is reused, the smallest actuation edit to validate first is to replace zero-width closure during the push with a fixed interior position setpoint, and hold that position at each finger using sufficient stiffness and a force cap. Keep the same Cartesian commands and check for both acceptable joint travel and repeatable rod contact before comparing estimators. If the existing coupled tendon cannot hold each finger inside its range under contact, use fixed fingers as a defined rigid pusher in the simulator and recalibrate the response fits. The saved B/C comparison remains paired for its original trajectories. Its yaw and history findings should not be generalized to a rigid or differently actuated pusher.

## Decision

**Close this rod iteration and skip the proposed fresh yaw-filter experiment.** Exact yaw identifies individual recoverable failures, especially case 31, but supplies no consistent release improvement with the saved fits. Corrected B and C tie on successful releases, and improving the simple yaw-response measurement rescues as many releases as it loses. The previous proposal required four net additional stable releases at the same 20-command, 2-second push cost. The fixed-fit exact-response diagnostic yielded one for C and zero for D, while contact sensitivity was comparable to the yaw errors. This leaves no evidence that a yaw-only change would clear that practical bar. Retain corrected B as the final-state reference for any reuse of this dataset.

## Reproduction

From `projects/active_robot_adaptation` with the existing pinned environment and source:

```sh
OPENBLAS_NUM_THREADS=1 /tmp/active-asid39/bin/python rod_balance/review_velocity.py
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl /tmp/active-asid39/bin/python rod_balance/review_fingers.py
```

The first command fits only the existing small regressions on the saved development split to reproduce the corrected predictions and score yaw substitutions. The second replays stored commands to recover physics-step timing and runs six actuation counterfactuals. Neither creates a new sample of rods. See the [per-case corrected scores and yaw substitutions](../results/published_runs/20261122T214828.954735Z_f896f1f094cc_rod_velocity_review_d5698762/summary.json) and [finger timing and actuator counterfactuals](../results/published_runs/20261122T214904.067900Z_f896f1f094cc_rod_finger_review_e948d850/summary.json).
