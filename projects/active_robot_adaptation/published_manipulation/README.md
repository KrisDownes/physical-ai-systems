# ASID author-code feasibility

The reports in this directory define the scope. Source is downloaded into `/tmp` and not vendored. The upstream tree has no top-level license; consult its authors before redistributing code. The Franka asset license is included in the upstream archive at `env/robot/assets/franka_assets/LICENSE.md` and is Apache 2.0.

From `projects/active_robot_adaptation` on Linux with EGL and `uv` available:

```sh
curl -fL https://codeload.github.com/WEIRDLabUW/asid/tar.gz/cf2cb670fb983f67cf2af156d4e17b4a690787e2 -o /tmp/active-asid-cf2cb670.tar.gz
printf '%s  %s\n' caa46b6e0925370d4b005eb7d574a7169cb7ef27eb615eb4f43c50470a35b826 /tmp/active-asid-cf2cb670.tar.gz | sha256sum -c -
mkdir -p /tmp/active-asid-src
tar xzf /tmp/active-asid-cf2cb670.tar.gz -C /tmp/active-asid-src --strip-components=1
uv venv --seed --python 3.9.25 /tmp/active-asid39
uv pip install --python /tmp/active-asid39/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r published_manipulation/asid-requirements-lock.txt
/tmp/active-asid39/bin/python -m pip check
```

The upstream multiprocessing launcher requires local sockets. A restrictive sandbox can reject its forkserver; use an appropriately authorized local execution environment. This 480-second cap is a feasibility smoke test, not the 250,000-step training run:

```sh
OPENBLAS_NUM_THREADS=1 /tmp/active-asid39/bin/python published_manipulation/asid_train.py --steps 560 --max-seconds 480 --seed 17
```

Evaluate the retained smoke checkpoint (the code resolves relative paths before entering the author source directory):

```sh
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/active-asid-mpl PYTHONPATH=/tmp/active-asid-src /tmp/active-asid39/bin/python published_manipulation/asid_evaluate.py --checkpoint results/published_runs/20261117T025228.231436Z_5961c1b14924_asid_train_9d430ce4/author_log/asid_rod/17/explore/policy.zip
```

Both commands create fresh unique directories containing configuration, dependency freeze, source snapshot and revision metadata. The exploration comparison uses simulation parameter truth only to set and score evaluation cases. Its metric is the authors' information reward, not a manipulation outcome. The checkpoint is a two-update local smoke model, not an author-released weight or a trained reproduction.
