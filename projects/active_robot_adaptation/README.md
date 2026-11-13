# Active robot adaptation

This project explores adaptation to changes in a planar two-link mechanism. All simulator, controller and evaluation code so far is agent-authored; `learner_kinematics.py` is learner-authored.

The analytical plant uses joint angles and increments in radians, tip position in metres in the planar base frame, and 0.1 s steps. Commands are clipped and a fixed fraction executes, subject to joint limits. This kinematic response is not a motor model: inertia, torque, contact and backlash are omitted. The affine camera maps base-frame position to camera coordinates; this is simulation, not physical validation.

The development comparison rotates the camera at a fixed step, without notifying either controller. All use camera-space tip and target, their own sent commands, and a common pre-change local calibration. The calibration is computed from simulator geometry at the initial configuration: it is an idealized calibration, not something the controller observes online. The fixed controller retains it; the online controller updates an action-to-observation Jacobian using recursive least squares. The sudden observation jump is included in the estimator's update and can corrupt it. The model-based controller additionally uses ideal nominal link lengths, response, camera calibration and a negative-elbow prior to infer its starting configuration from the first observation, then dead-reckons from its commands and updates its Jacobian. It does not receive joint-state observations; its extra model knowledge makes it a privileged reference, not a sensor-matched learned estimate. All use the same damped inverse and command bounds. The evaluator alone holds simulator state and change schedule. Post-change target reachability is checked using the radial workspace, valid for the default full joint limits and invertible camera.

This is one deterministic, hand-selected development example, not a tuned benchmark or generalization claim. Recovery means five consecutive samples inside 0.015 camera units; the reported time is the time until the fifth sample, relative to the change. Integrated error sums all post-change samples, including recovery and any exploration, scaled by 0.5 camera units and step time. Failed runs are retained with a null recovery time. The horizon continues for all 140 post-change steps. Controller and plant saturation are logged separately, as are sent and executed increments.

From this directory, after installing the locked requirements in the project-local Python 3.12 environment:

```bash
.venv/bin/python -m pip install -r requirements.lock.txt
PYTHONPATH=src MPLCONFIGDIR=/tmp/active-adaptation-mpl .venv/bin/python scripts/smoke.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/active-adaptation-mpl .venv/bin/python scripts/evaluate.py
PYTHONPATH=src MPLCONFIGDIR=/tmp/active-adaptation-mpl .venv/bin/python scripts/paired_development.py
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The two development commands write compact summaries and raw traces plus headless plots under `results/development/` and `results/paired_development/`. The paired sweep has seven camera angles, one starting pose and target, and no noise; its configured seed is unused because the run is deterministic. M1 still needs broader paired schedules, fair tuning and a learning review; the fixed probe and active methods come later. See STATUS.md.
