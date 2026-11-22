# Active manipulation: current research index

**Current direction (2026-11-17):** build and test our own interaction-based manipulation mechanisms. Published reproduction, a released checkpoint, and novelty claims are not prerequisites. This supersedes prospective instructions in older handoffs and reports; those reports remain historical evidence. SO-101 remains an integration testbed. Guided-slider conclusions apply only to its tested conditions.

| Asset | What is runnable / evidence | Open limitation |
| --- | --- | --- |
| [SO-101 simulator](so101/README.md) | Vision/calibration and kinematic manipulation testbed; [transfer results](so101/TRANSFER_RESULTS.md). | Not a validated digital twin. |
| [Bottle](so101/BOTTLE_RESULTS.md) | Rigid miniature bottle and fill/friction baselines with saved trials. | Enclosure displacement, empty/wet feasibility and comparator joint limits unresolved; not reopened automatically. |
| [Shared model](so101/SHARED_RESULTS.md) and [decision audit](so101/DECISION_AUDIT.md) | History/current predictors and guided push/pull. Some goal-gated actions bypassed prediction; resistance differences did not change tested action choices. | Does not rule out adaptation elsewhere; do not keep tuning these guided conditions. |
| [ASID feasibility](published_manipulation/RESULTS.md) | Author rod scene, 2D Franka actions, pose observations and FIM exploration run locally. | No downstream balancing pipeline/checkpoints released; FIM proxy is not a manipulation outcome. This is an implementation constraint, not a research rejection. |
| [Independent rod experiment](rod_balance/RESULTS.md) | Author pushing interface plus our explicitly idealized physical support release and matched models. | No continuous grasp/transport or validated physical transfer; see its hypothesis log and traces. |

## Mechanisms and assumptions

- [ASID, Sec. 3 and simulation experiments](https://arxiv.org/html/2404.12308): exploration seeks sensitivity of predicted observations to latent physical parameters; identification then informs a downstream action. Assumes an informative interaction, an adequate model and simulator-only parameters for its training reward. Our interpretation: a simple observed push-response estimator is a cheaper starting point; it need not use ASID's reward. The author's code revision and license status are in [selection](published_manipulation/SELECTION.md); upstream code is not copied.
- [RMA², method](https://arxiv.org/html/2312.04670): a learned adaptation module infers a latent from action/observation history while policy weights remain fixed during deployment. Assumes training covers useful latent variation and sensor history identifies it. Our interpretation: a fixed context encoder could later replace or complement the explicit estimator after a useful decision is established.
- [Poke and Strike](https://arxiv.org/html/2509.00178): a poke supports system identification for a later task. Assumes the exploratory action is feasible and its information affects the downstream action. Do not claim novelty merely for combining history and probing.

The near-term decision is whether interaction-derived observations improve support placement enough to justify their cost, relative to prior/current-state estimators. Only after an effect survives independent starts and negative controls should a reusable estimator be frozen and tested on a second interaction. Transfer of an estimator is not whole-task transfer. No physical robot, paid services or purchases are authorized.

**2026-11-22 rod decision:** [one 48-case matched iteration](rod_balance/ITERATION_RESULTS.md) found lower average error from history, but did not establish a stable-release improvement over complete current-state feedback. The finite-difference joint-velocity comparator correction is explicitly post hoc. A same-push yaw-observation precision check is the single next proposed test; close the branch if its prediction fails. Prior cases remain development data.

**Rod iteration closed.** The [saved-data review](rod_balance/VELOCITY_REVIEW.md) reproduced a 17/48 tie for corrected final-state and history models. Exact yaw rescued individual misses but gave at most one net additional release with the fixed history fit. Finger actuation changed rod response on the scale of yaw observation error. The evidence does not warrant the proposed fresh yaw-filter test.
