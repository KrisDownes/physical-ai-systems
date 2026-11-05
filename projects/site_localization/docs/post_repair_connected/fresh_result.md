# Search-order isolation and connected-layout transfer

## What was tested

The [matched handoff diagnostic](matched_result.md) established a scoped causal effect on the opened wrong-asset map: with identical acquired map, physical checkpoint, estimated AMCL initialization and budgets, visibility ordering reached the eastern camera station about 100 s and 3.20 m of inspection travel before nearest ordering. Both first decoded ID7 from nearly the same 3.3 m, ~41 px, **unreadable-gauge** view. Only the earlier visit left enough travel to approach and make a valid reading. The reset was an evaluator-controlled diagnostic, not continuous autonomy. The moved checkpoint passed under both policies; both absent checkpoints withheld completion, though the visibility policy spent more time and distance searching. [Full paired numbers and startup failures](matched_result.md) · [Raw checkpoint analysis](../../results/matched_visibility_20261027/analysis.json).

The prospective [fresh protocol](fresh_protocol.md) was recorded before scoring. The new connected service-suite layout uses a central 1.5 m doorway to the eastern room and a separate west screen; the opened development layout had a northern east-room doorway. [Evaluator-only preflight](../../results/fresh_visibility_20261027/evaluator_preflight/feasibility.json) checked connected free floor, a collision-clear path to each 1.1 m inspection witness, and usable rendered gauge imagery. Those witness poses, truth map and exact changed geometry were never method inputs. The [133-file pipeline freeze](../../results/fresh_visibility_20261027/pipeline_freeze.json) and [48-file scenario freeze](../../results/fresh_visibility_20261027/scenario_freeze.json) were verified after execution with no mismatches. Within each case, method templates differ only by `search_selector`; each arm then independently acquired its own map. All eight attempts ran sequentially, completed scoring, and reaped their processes. [Execution ledger](../../results/fresh_visibility_20261027/execution.jsonl) · [All process/attempt events](../../results/fresh_visibility_20261027/attempts.jsonl).

## Complete uninterrupted missions

| New-layout case | Nearest: physical result; captures; total time/distance | Visibility gain: physical result; captures; total time/distance |
|---|---|---|
| ID7 unchanged at old site | **Complete**; 3; 153 s / 12.21 m | **Complete**; 3; 169 s / 14.62 m |
| ID7 moved east | **Complete**; 18; 284 s / 18.36 m | **Complete**; 18; 254 s / 18.21 m |
| ID8 old, ID7 east | **Complete**; 18; 274 s / 18.51 m | **Complete**; 18; 303 s / 20.47 m |
| ID7 absent, ID8 old | Honest `search_budget_exhausted`; 32; 344 s / 19.26 m | Honest `search_budget_exhausted`; 32; 351 s / 19.67 m |

Six of six target-present missions had a valid **final claimed capture**, correct ID7, safe 0.9–1.3 m inspection pose, heading/dwell/reading compliance and complete acquisition-to-inspection handoff. Both absent missions decoded ID8 at the old location and withheld completion. There were **zero false completions, wrong-asset acceptances, contacts, interventions and resets**. Minimum clearance across all eight was 0.306 m. Acquisition maximum evaluator position errors ranged 0.036–0.183 m; handoff seed position errors ranged 0.005–0.088 m. No acquisition or handoff failed. [Per-mission raw analysis](../../results/fresh_visibility_20261027/analysis.json) includes mapping/localization error, predicted map area, actual camera captures, station order, first ID7 detection, approach costs and physical outcomes. Camera frames and inspection visits are correlated within each mission; these are eight runs in **one environment with four case conditions**, not eight independent site deployments.

The two absent runs make no certified absence claim. They exhausted the declared search-position/capture budget with no ID7 evidence; unobserved or unreadable viewpoints could still exist. Their outcome tests honest reporting under a bounded search, not exhaustive detection coverage.

## What the mechanism did and did not transfer

On the fresh moved and wrong-asset cases, nearest chose the central doorway station `(0,0)` after the old view. Visibility gain chose `(1.5,0)`. Both stations produced decoded ID7 at capture 15 and then valid inspection. In the wrong-asset case, the independently rendered central view had full visible gauge face at about 113 px and the east view about 119 px; neither station was yet within the required 0.9–1.3 m physical inspection range. The gain method predicted more newly visible **2D map-free area** at `(1.5,0)` than `(0,0)` (wrong-asset score 3.35 versus 3.30 area/(path+1 m)); the small score margin does not represent a measured difference in marker detectability. The actual central camera view was sufficient to identify ID7. In the wrong-asset pair, the gain station shortened the *subsequent map-planned approach* (1.54 versus 2.56 m) but took longer/more motion to search and independently acquired its map later. It finished 29 s and 1.96 m behind nearest overall. On the moved pair, gain finished 30 s earlier but only 0.15 m shorter overall; its acquisition finished 15 s earlier, so that total-time difference cannot be attributed entirely to search ordering. [Gain choice trace](../../results/fresh_visibility_20261027/mission_wrong_asset_gain/inspection/method.jsonl) · [Nearest center-view trace](../../results/fresh_visibility_20261027/mission_wrong_asset_nearest/inspection/method.jsonl).

First correct ID7 detection on the fresh wrong-asset case occurred at 243.1 s and 6.56 m of post-handoff physical travel for nearest, versus 284.8 s and 7.92 m for gain. The independently measured remaining 18 m travel allowances were 11.44 and 10.08 m, both sufficient for the respective planned approaches. On the fresh moved case, gain detected at 236.3 s and 8.81 m post-handoff, while nearest detected at 246.4 s and 6.47 m. These timings include different acquisition/handoff start times; the distance figures isolate inspection-stage travel more directly but still reflect independently acquired maps.

The score's prediction is geometric 360° **map-plane line-of-sight area** within 0.6–5 m, with eight later 60° camera captures at each station. It does not model vertical occlusion, panel facing, readable pixel size, or whether the already sufficient central view contains the intended marker. The fresh outcome does not prove its raycast is geometrically wrong; it shows that extra predicted free-space area was unnecessary for this asset and did not consistently reduce whole-task cost. The opened-map rescue remains real under the matched checkpoint, but a broad improvement in complete-mission success did **not** appear here. No threshold, budget policy, perception or viewpoint score was changed after scoring.

## Decision and reproduction

The conventional pipeline now demonstrates continuous sensor-driven mapping, estimated-pose handoff, visual identity, safe approach, a valid synthetic inspection reading, and honest bounded non-completion on a second connected layout. Known initial dock assistance, ideal calibration, unique attached marker, synthetic camera/reader and one layout remain major limits. **Simulation success does not validate physical camera transfer, field reliability, or product demand.** This is engineering progress, not originality: [semantic next-best-view research](https://arxiv.org/abs/2404.16507) already addresses limitations of visibility-only search, and practical inspection systems such as [AutoInspect](https://www.robots.ox.ac.uk/~nickh/papers/staniaszek2024autoinspect.pdf) use operator-commissioned task locations. A [separate budget-feasibility replay](budget_replay_protocol.md) showed that a simple reserve would reject the opened nearest policy's infeasible late east leg but would not find ID7. A research claim would require a consequential residual after a strong camera/object-aware, budget-aware comparator with equal information. The next bounded comparison should examine whether such an established method improves *complete task cost or success* under camera occlusion or tight approach reserve, using fresh layouts and actual observed image geometry. Do not optimize the current 2D score on these opened outcomes.

From repository root, inspect saved results without a simulator:

```bash
/usr/bin/python3 projects/site_localization/research/post_repair_connected/analyze_matched.py
/usr/bin/python3 projects/site_localization/research/post_repair_connected/analyze_fresh.py
```

For a new development reproduction of one **continuous** mission, source ROS Jazzy and choose a new output path:

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 projects/site_localization/research/continuous_asset/run.py \
  --template projects/site_localization/results/fresh_visibility_20261027/templates/wrong_asset_gain \
  --out projects/site_localization/results/fresh_visibility_reproduction_01 \
  --slam-params projects/site_localization/research/constraint_repair/slam_params_radius_1p5.yaml \
  --seed 7501 --wall-limit 800
```

For the matched **reset-based diagnostic**, use `run_matched.py --single wrong_asset gain --suffix _repro01 --manifest manifest_v3.json` after sourcing ROS. It uses the archived checkpoint and writes a separate attempt, not an uninterrupted mission. ROS/Gazebo requires permitted local socket access. The [tutor exercise](fresh_tutor_exercise.md) traces the decisive viewpoint and approach-budget arithmetic.
