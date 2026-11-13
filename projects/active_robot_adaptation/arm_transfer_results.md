# Arm transfer pilot results

## Held-out negative-elbow cases

Four nominal validation starts and targets reached the 0.015 camera-unit tolerance for five consecutive observations. The subsequent evaluation used four different starts/targets, two held-out single-joint gain values and two held-out camera rotations. It was frozen after development tuning. Each method ran the same 20 case/noise schedules; no tested target was unreachable. All four methods succeeded in all 20 episodes. These small, favorable kinematic cases do not establish general reliability.

| Condition | Episodes per method | Nominal | Online Jacobian | Global correction | Global + GP |
|---|---:|---:|---:|---:|---:|
| No change | 4 | 0.0942 | 0.1048 | 0.0945 | 0.1045 |
| Actuator gain | 8 | 0.1202 | 0.1258 | 0.1108 | 0.1156 |
| Camera rotation | 8 | 0.0849 | 0.0986 | 0.0848 | 0.0861 |

Entries are median integrated normalized post-switch error in simulated seconds. Median recovery times on gain cases were 1.15, 1.20, 1.10 and 1.05 simulated seconds respectively; the same group had median peak target errors of 0.1342, 0.1342, 0.1342 and 0.1343 camera units. All cases have 120 actions including the 20 pre-switch actions. The GP was slower on this CPU: median total controller-computation time on gain cases was about 0.030 s versus 0.005–0.008 s for the other methods. This is measured process wall time over 120 actions, not embedded or real-time latency. Full episode metrics, recovery curves and traces are in `results/arm_transfer/`.

The global method outperformed this tuned Jacobian implementation on median gain-case error; the GP did not improve on the global-only median. The nominal controller also succeeded in all cases. The online baseline has a simple local RLS model and a limited tuning grid; a stronger configuration-aware or better-tuned baseline could change the ranking. The evaluation target switches at the disturbance time, and the analysis has no confidence interval. These results are a pilot, not a general performance claim.

## Branch ambiguity diagnostic

After the frozen evaluation, each no-change start was mirrored to the opposite elbow configuration while preserving initial tip position, target history, noise and action budget. With the positive elbow, nominal, online, global and GP succeeded in 2/4, 4/4, 4/4 and 3/4 cases; their median errors were 4.4433, 0.1937, 0.3844 and 1.9040. The corresponding negative-elbow group succeeded 4/4 for every method. These are four matched diagnostic cases, not a held-out estimate. The camera-only negative-branch prior is a demonstrated source of nominal-model mismatch, but the current results do not isolate whether GP failures arise from its model, inversion, data selection, or interactions among them. See `branch_stress.json` and `branch_stress.png`.

## Tinygrad learned controller

Tinygrad 0.14.0 differentiated `x²` at 3 as 6 and trained a Gaussian policy-gradient bandit toward its known optimum, ending at 1.006. An initial 10-update arm workload took 10.50 s on Tinygrad's Python backend and 2.23 s including startup/compilation on CPU with a locally extracted clang 18.1.3. The CPU run trained three seeds for 300 updates × 128 one-step contexts; wall times were 16.04, 14.17 and 14.44 s. This is a contextual-bandit policy-gradient controller trained on kinematic, negative-elbow samples; it is not a long-horizon or torque-control policy. Samples use camera position and requested displacement, and train bounded joint commands from one-step displacement reward. The frozen weights are stored under `results/arm_transfer/tinygrad/`.

Each frozen policy was evaluated on the same 20 cases with and without a digital global displacement-correction layer. Every policy-condition pair succeeded in 20/20 episodes. Across seeds 1–3, median integrated errors without correction were 0.1490, 0.1480 and 0.1811; with correction they were 0.1403, 0.1354 and 0.1519. For camera-change cases, correction was slightly worse for seeds 1 and 2. The policies and correction were not tuned on these cases, but the cases have already been inspected in the earlier controller pilot; this is not a fresh independent final benchmark. The correction also changes no-change behavior, so the difference cannot be attributed solely to fault recovery. Evaluation uses frozen Tinygrad-trained weights in an equivalent NumPy forward pass; adaptation remains digital. These outcomes do not establish physical performance.

## Candidates to discuss

1. **Add branch information to every method**, for example joint encoders or an observed elbow marker. It addresses the demonstrated identical-tip/different-configuration ambiguity. Prediction: positive-elbow nominal and GP failures decrease under the same paired budget. Disproof: add the channel equally to all methods on matched branch cases and find no reduction in failures or excursions. Configuration-aware kinematics and visual servoing are established approaches, including [uncalibrated image-based visual servoing](https://www.iri.upc.edu/files/scidoc/1412-Uncalibrated-image-based-visual-servoing.pdf); no novelty is claimed.
2. **Maintain multiple camera-only branch models and update their relative plausibility from commanded motion.** It addresses ambiguity without a new sensor, but requires informative movement and may be unsafe during uncertainty. Prediction: fewer positive-elbow failures than a single negative-branch transfer model at equal action cost. Disproof: compare against the current online Jacobian on the same paired branch cases and reject if failure or integrated error is not lower. Multiple-model adaptive control for manipulators is prior art: [adaptive control using multiple parameter models](https://ap.isr.uc.pt/archive/568.pdf). No novel mechanism is established.

These are starting points for joint selection, not validated explanations or a claim that exploration is required. Before choosing a mechanism, inspect GP residuals, conditioning, and response to the camera jump; use new held-out cases for the next comparison.
