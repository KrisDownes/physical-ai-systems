# Rover handoff — offline review completed 2026-09-10


## Latest offline review (no new simulator/model execution)

Evidence supports potentially avoidable backtracking at decision4: selected route
7.50 m / 8 frontier cells versus eligible alternative2.75 m / 88 cells, using only
its exact onboard map/pose. Actual accepted-selection leg traveled6.349 m and
failed bounded replanning. Sector clearance does not prove the alternate blocked;
no counterfactual mission-time savings established. Other longer choices often
traded distance for component size and cannot be labeled waste from distance alone.

Decisions6/8 repeat (5.798242681,.353075990), not a second arrival. Goal6 recovery
at272.025 sim s imposed30 s cooldown; request9 at302.130 followed its expiry by
.105 s. It was the only supplied candidate, unvisited; intervening goal remained
excluded. Map origin/grid unchanged, ten cells changed, SLAM correction moved
~.102 m. Retry permitted by memory policy; usefulness unresolved. Local retargets
and eventual failure promotion were robotics responsibilities.

New `HIERARCHICAL_INPUT_SPEC_v2.md`: retained `cells` compatibility, defined as
frontier_component_cell_count, NOT free-space area/coverage/traversability.
Changed `scripts/hierarchical_driver.py` only to append definition/version to prompt
and record input_spec_version; `scripts/test_hierarchical.py` adds one focused
schema check. No navigation/safety/model/evaluator changes. Historical baseline,
contracts and prompts unchanged; future prompt differs explicitly from baseline.

Artifacts: `experiment_runs/offline-luna-decisions-20260910/` contains report.md,
decision_table.md (nine rows), decisions.json (all costs/events/history), audit.py,
verify.py, verification.json, before/after hashes, field_clarification.patch,
pre-change driver copy, relevant exact image exports, next_experiment.json.

Checks run (offline only):
```bash
source /opt/ros/jazzy/setup.bash
python3 experiment_runs/offline-luna-decisions-20260910/audit.py
python3 experiment_runs/offline-luna-decisions-20260910/verify.py
PYTHONPATH=scripts:src/rover_exploration python3 -m pytest -q scripts/test_hierarchical.py -k 'frontier_size_definition or selected_destination_controls_path or response_timeout_and_model_disabled'
```
3 tests passed /13 deselected; nine-map/route/candidate joins, memory replay and
protected-code/prompt checks passed. An offline assertion initially included
route cost in repeated-destination identity; corrected to identity fields (poses
and route costs legitimately differ). No Gazebo, benchmark episode, ROS node or
rover-model call occurred. Saved bags were only deserialized; no live checks.

Recommended next experiment ONLY: one frozen nine-snapshot Luna/low semantic replay
with v2 wording, ≤9 external turns/600 wall s, no retries. Prediction0/9 explicit
free-space interpretations of cells (baseline5/9). Plan in next_experiment.json
has enable_model=false; it is NOT launched/authorized and tests semantics, not
physical savings. This supersedes the prior seed-repeat recommendation.

## Successful live baseline / authorization

Hierarchical live comparison and ONE explicitly user-authorized Luna restart are
finished. **Do not launch another experiment without new authorization.** No model
substitution, automatic retry, classical rerun, runtime refactor or prompt change.
All original artifacts and unrelated Git changes preserved. No AGENTS.md found in
workspace/ancestors. Workspace:
`~/physical-ai-systems/projects/active_exploration_rover/ros2_ws`.

Latest report: `experiment_runs/hierarchical-luna-restart-20260910/report.md`.
Machine-readable primary comparison: same root `comparison.json`; includes saved
classical, restarted Luna, and original quota-limited Luna separately.

| Metric | Saved classical | Restarted hierarchical Luna |
|---|---:|---:|
| Original evaluator | FAIL: map→odom jump .30015 m > .25 m | PASS |
| Native completion | yes | yes |
| Wall / simulation s | 367.40 / 238.58 | 479.67 / 336.74 |
| RTF | .649 | .702 |
| Known rectangular grid | 97.75% | 98.63% |
| Frontier components ≥5 cells | 0 | 0 |
| Path | 22.16 m | 31.78 m |
| Model turns / accepted / rejected | 0 / N/A / N/A | 9 / 9 / 0 |
| Local assignments / arrivals | 15 / 1 | 11 / 1 |
| Model waiting | 0 | 61.66 s (12.86%) |
| Safety / controller-stall s | 6.05 / 7.36 | 3.85 / 6.43 |

Luna actual model gpt-5.6-luna/low, one persistent session. All nine turns received
matching saved camera/map image payloads + valid lidar/estimated pose/candidates.
Nine accepted destination choices caused continuous local motion (e.g. turn 3→4:
858 controller updates, 10.293 m, no extra selector call). One arrival at 0.2175 m
(<0.25 criterion); goals can become obsolete without arrival. Two extra local
retargetings explain 11 assignments vs nine model goals. Two recovery events.
No stale/rejected response used the turn budget. No restart transport/quota error.
Model round-trip median/p95 6.031/10.210 wall s, includes transport/internal work;
executive model-wait includes snapshot/revalidation overhead, not pure inference.
Usage: 99,406 input (65,024 cached), 1,792 output (1,346 reasoning), 101,198 total.
No provider response ID or internal retry/compaction notification exposed; raw
thread/turn/message IDs retained. No dollar costs inferred.

Both had one permanently failed region warning. Luna left 37 raw frontier cells
in 22 subthreshold components; classical 18 in 11. Known-grid is NOT accessible
coverage. No contact sensor / collision-free claim. Luna passed all original hard
gates and recorded 6.617 sim s after completion: zero active velocity/path/truth
motion. Final /cmd_vel zero; all six owned groups exited0/group_gone=true.
Host process audit confirms no owned simulator/ROS/recorder/runner/model worker.
Preexisting MCP processes1275/1276 remain untouched (host rebooted since old pair).

## Exact commands / paths

Already executed original pair:
```bash
./scripts/run_hierarchical_comparison --enable-model --output experiment_runs/hierarchical-luna-pair-01
```
Original Luna stopped on explicit `usageLimitExceeded`, NOT interruption or
SIGTERM-inferred quota. Four turns / two accepted / one stale / zero arrivals,
83.858 wall s, 55.36 sim s, 4.290 m, 79.65% known grid. Safe cleanup preserved.
User then explicitly requested restarting the second run. Executed ONCE:
```bash
source /opt/ros/jazzy/setup.bash
./scripts/run_hierarchical_pilot --hierarchical-selector llm --enable-model --wall-budget 600 --output experiment_runs/hierarchical-luna-restart-20260910
```
Episode directories:
- Primary classical: `experiment_runs/hierarchical-luna-pair-01/classical/classical/`
- Original quota Luna: `experiment_runs/hierarchical-luna-pair-01/llm/classical/`
- Successful restart: `experiment_runs/hierarchical-luna-restart-20260910/classical/`
Nested `classical` is the legacy shared-controller slot, NOT Luna's selector label.
Use manifest.selector and hierarchical_model_metadata.json; versions.json has
legacy model:null/method:classical fields for this runner slot.

Each episode: `bag/`, results.json, original_benchmark.json, bag_verification.json,
manifest.json, cleanup.json, executive.jsonl, events.jsonl. Luna: driver_rpc.jsonl,
driver_stderr.log, goal_observations.jsonl, hierarchical_model_metadata.json.
Parent source/, source_hashes.json, versions.json identify executed code.
Restart root additionally: report.md, comparison.json, comparison_configuration.json,
observation_delivery_audit.json, processes-after-run.txt, initial/latest image
exports, offline analysis scripts. Auto-generated runner report retained as
runner_generated_report.md; its generic diagnostic wording is incorrect for this
live run and superseded by report.md. No runtime fix made afterward.

Offline regeneration (no model/simulation):
```bash
python3 experiment_runs/hierarchical-luna-restart-20260910/analyze_comparison.py
python3 experiment_runs/hierarchical-luna-restart-20260910/write_report.py
```

## Implementation / verification / limitations

Contract HIERARCHICAL_BENCHMARK_CONTRACT.md v2, v1 preserved. 600 episode wall s,
≤20 externally submitted decision turns, no application retries/substitutions;
NOT a guaranteed internal inference/token/monetary cap. ROS_DOMAIN_ID87, kd_world,
spawn(0,0,.02,0), fresh SLAM/Gazebo per episode, GUI and bags automatic.
LLM selects one supplied online-map approach; shared custom ExplorationPolicy/BFS,
PathFollower/ObstacleGuard + steady-clock gate execute/recover locally; NOT Nav2.
Same candidate validation, sensors, motion limits and evaluator as classical.
Direct-control benchmark remains separate and historical (different watchdog).

Motion/selector/safety/driver hashes match successful mock
`experiment_runs/hierarchical-route-a1-20260909/`. Original pair preflight.json and
changes_since_mock.patch document only prior reporting/budget/launcher changes.
Restart source hashes and package versions exactly match saved classical/original
Luna; four-package build passed before restart. Reused previous focused tests;
no unnecessary new live runs. Only offline analysis/report and handoff were added
this turn. Git base c52a3ece320f2c9ed51f6b9a83cbc2b4842a8c57; dirty-tree hashes matter.

Pilot conclusion: live Luna destination selection + classical continuous control
can meet this benchmark with nine decisions. It was slower/longer than classical;
no controlled reduction in waiting vs direct-control established. One pair, host
reboot/RTF differences and nondeterministic SLAM limit superiority claims.
Luna calls candidate `cells` free-space support although it is frontier-component
size; sector clearance does not establish reachability. Shared planner validation
provided the operational check. The subsequent offline input-v2 clarification is described above; navigation remains unchanged.

Next experiment recommendation is now the bounded semantic replay above; no live run is authorized.
