# Reproduce the continuous mission

Run from `/home/krisd/physical-ai-systems`. Existing ROS Jazzy, Gazebo Harmonic, SLAM Toolbox, official AMCL library build and vision virtual environment are used; no install or purchase is needed. Check no other session is simulating. The runner refuses an existing output directory or another Gazebo server, uses ROS domain216 with a unique Gazebo partition, and cleans up only its own processes.

Inspect the preserved results without starting physics:

```bash
/usr/bin/python3 projects/site_localization/research/continuous_asset_analysis.py \
  projects/site_localization/results/continuous_asset_20261025
```

This recreates derived CSV/diagnostics and checks hashes; it does not change raw logs, labels, reader, protocol or freeze. Optional demonstration reconstruction:

```bash
MPLCONFIGDIR=/tmp/site-localization-mpl /usr/bin/python3 \
  projects/site_localization/research/continuous_asset_analysis.py \
  projects/site_localization/results/continuous_asset_20261025 --demo fresh_moved
```

Run a **new development reproduction**, retaining the original outputs:

```bash
source /opt/ros/jazzy/setup.bash
source projects/site_localization/vendor/setup_amcl.bash
/usr/bin/python3 projects/site_localization/research/continuous_asset/run.py \
  --template projects/site_localization/results/continuous_asset_20261025/scenarios/development/moved \
  --out projects/site_localization/results/continuous_asset_20261025/reproduction_moved_01 \
  --seed 7301 --wall-limit 800
```

The same command with `--return-only`, an unused output directory and the development unchanged/moved template reproduces the **separate old return-required contract**. It does not continue to inspection. All old V1/V2 attempts and their source snapshots remain available. ROS/execution randomness and callback scheduling mean a seed alone does not imply bitwise reproduction.

Historical fresh protocol commands, **already executed once**:

```bash
/usr/bin/python3 projects/site_localization/research/continuous_asset/evaluate.py \
  projects/site_localization/results/continuous_asset_20261025
source /opt/ros/jazzy/setup.bash
source projects/site_localization/vendor/setup_amcl.bash
/usr/bin/python3 projects/site_localization/research/continuous_asset/evaluate.py \
  projects/site_localization/results/continuous_asset_20261025 --execute
```

The first command now refuses to overwrite `final_freeze.json`; the second refuses to replace existing mission directories. Do not delete those safeguards to manufacture a fresh result. Any new development run must get its own directory and remains development evidence. The opened fresh cases are now development data for subsequent changes.

Implementation: `research/continuous_asset/run.py` orchestrates a single world; `explore.py` selects live frontiers; `buffered_planning.py` separates routing margin from hard-safe connectors; `control.py` checks motion commands; `inspection.py` retains frozen identity/inspection behavior. `score.py` and `inspection_judge.py` evaluate truth separately. `handoff.json` is the estimated-state transfer record. `stages.jsonl`, `run.json`, `acquisition/exploration.jsonl`, `inspection/method.jsonl` and `judge.jsonl` provide the causal trace.

Current code/config identity is the76-file `final_freeze.json`, not Git HEAD alone: the project is untracked in this workspace. Each attempt also has its own source/config hashes and Python source snapshot. The audit found no changes among3553 protected prior files or76 frozen files, and all eight attempted executions were scored and all owned processes reaped.
