# Planar baseline checkpoint

The agent-authored two-link plant is kinematic: 0.30 m and 0.20 m links, radians, ±0.1 rad sent commands, 0.8 nominal response and 0.1 s steps. The affine camera maps tip position to camera units. Controllers see camera tip and target, their own commands and time; simulator joints, change labels and executed motion are evaluator-only. The original fixed and RLS Jacobian controllers use an idealized initial analytical calibration. The later transfer pilot uses a capped 0.025-unit desired camera displacement, a nominal negative-elbow inverse, global RLS behavior correction, and a local GP residual. The online baseline is tuned only over a limited grid. See `arm_transfer_interface.md` and `arm_transfer_results.md` for attribution, interface, assumptions and outcomes.

Development configurations and seed schedules, the frozen 20-case evaluation and 80 episode metrics, and compact branch results are in `results/arm_transfer/`. All methods succeeded 20/20 negative-branch pilot cases. Median normalized integrated errors on eight held-out gain cases were 0.1202 nominal, 0.1258 online, 0.1108 global, 0.1156 GP. The separately inspected opposite-elbow diagnostic succeeded 2/4, 4/4, 4/4 and 3/4 respectively. The latter reused evaluated starts, so is not a fresh held-out study. The disturbance coincided with a target switch; no uncertainty interval, physical actuator dynamics, real sensor latency or physical demonstration was established.

Three Tinygrad 0.14.0 contextual-bandit policies were trained on the planar kinematic task, not SO-101. Frozen weights, per-seed training results and compact evaluation summaries are retained. All policies succeeded 20/20 with and without global correction on cases already inspected in the earlier pilot. Their medians without correction were 0.1490/0.1480/0.1811 and with correction 0.1403/0.1354/0.1519. No long-horizon or torque-control RL result is established. `learner_kinematics.py` is learner-authored and still needs explicit trigonometric imports before independent execution; it is not used by the package.

The FLAIR paper's table was reanalyzed using authors' 20-run archived data; three new local examples were unpaired single runs. Both are distinct from a full 20-run reproduction. The archived angular-unit convention remains unresolved, and the example's generated time advances ten times more slowly than physics. The arm transfer is an independently authored simplification, not the authors' implementation. See `flair_findings/README.md` and the committed source links/checksums.

From this project directory, using Python 3.12 and dependencies pinned in `requirements.lock.txt` and `requirements-tinygrad.txt`:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-tinygrad.txt
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
PYTHONPATH=src MPLCONFIGDIR=/tmp/active-adaptation-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/arm_transfer_experiment.py eval
PYTHONPATH=src MPLCONFIGDIR=/tmp/active-adaptation-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/arm_branch_stress.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/tinygrad_arm_evaluate.py
```

The evaluation commands overwrite generated outputs in `results/arm_transfer/`; preserve the archived summaries before rerunning if needed. Raw traces and large local diagnostics are intentionally not part of this checkpoint; scripts regenerate them. Tinygrad CPU training additionally needs a working clang compiler; the prior temporary compiler was not part of the environment lock. Timing figures depend on the host and are not expected to reproduce exactly.
