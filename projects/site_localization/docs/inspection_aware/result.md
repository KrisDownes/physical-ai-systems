# Camera-aware search comparison: completed development and fresh missions

The [prospective protocol](protocol.md) compared nearest station, the frozen
2-D map-visibility selector, and a discrete Bayesian camera/object search
selector. The new arm is an engineering adaptation of the established sensor
planning formulation of [Ye and Tsotsos (1999)](https://research.ibm.com/publications/sensor-planning-for-3d-object-search),
not an original algorithm. It maintains a uniform belief over free map cells
and four possible asset-face directions, lowers belief after actual negative
ID7 images, and ranks fixed camera stations by expected ID7 detection per
travel with an approach-budget feasibility check. It never receives the moved
asset pose or orientation. [Implementation](../../research/inspection_aware/belief_search.py)
· [policy wrapper](../../research/inspection_aware/inspection.py).

We directly inspected [Yu and Chen's semantic next-best-view paper](https://arxiv.org/html/2404.16507).
Its RGB-D/segmentation/3-D TSDF reconstruction setting and ROI reconstruction
metric do not instantiate this planar marker-and-gauge task; it is relevant
prior art, not a drop-in comparator. A closer recent method,
[Taioli et al. (2024)](https://iris.polito.it/retrieve/2b0871ef-9313-4066-b0b6-80c4ab8f31a9/Unsupervised_Active_Visual_Search_With_Monte_Carlo_Planning_Under_Uncertain_Detections.pdf),
uses a 2-D floor-map belief and uncertain detections, but also RGB-D target
proposals and POMCP. Our small greedy comparator does not establish limits of
that stronger approach.

## What was held fixed

All policies used the acquired 2-D map, the same 3×3 station lattice and eight
headings per station, old commissioned pose, known initial dock, 1920×1080
60° camera, 16 cm AprilTag ID7, frozen detector/reader, navigation and
inspection controller, 18 m/420 s inspection limit, 35 m whole-task limit,
four station and 36 capture limits. The new arm's only online additions were
negative-image belief updates and station scoring; its budget gate is part of
the treatment. It assumed a 0.75 decode probability for geometrically
eligible views, **uncalibrated** against real camera data. The 2-D ray ignores
elevated occluders. It predicts ID detection, while the unchanged reader and
independent physical judge determine usable measurement and completion.
Its budget test uses Euclidean distance to a hypothesized 1.1 m stand-off
plus 1.4 m margin, a lower-bound approximation rather than a guaranteed
collision-clear approach route.

The old [matched checkpoints](../../results/matched_visibility_20261027/analysis.json)
supplied byte-identical acquired maps and an evaluator-created same physical
restart for each arm. No truth pose entered AMCL or the selector. These are
inspection-only diagnostic resets. One new-arm startup failed before useful
physics because sandboxed ROS bridge could not call `getifaddrs`; it is retained
in the [attempt ledger](../../results/inspection_aware_20261028/attempts.jsonl)
and was rerun outside the sandbox without changing the method.

| Opened checkpoint | Nearest | 2-D visibility | Camera belief |
|---|---|---|---|
| ID8 old, ID7 moved | **Failed**: ID7 first at 243 s / 15.05 m, approach budget exhausted | **Complete**: 143 s / 11.85 m | **Complete**: 183 s / 12.06 m |
| ID7 moved | Complete: 214 s / 10.79 m | Complete: 128 s / 9.25 m | Complete: 161 s / 9.35 m |
| ID7 absent | Withheld; 13.55 m | Withheld; 17.50 m | Withheld; 13.55 m |

Time/distance in the first two rows are to first decoded ID7, measured by
the independent analyzer. On the opened wrong-asset case, visibility's earlier
east-station visit remains the strongest observed search result. Camera belief
chose north then east and completed, but did not improve detection cost over
visibility. [New arm raw analysis](../../results/inspection_aware_20261028/analysis.json).

## Fresh uninterrupted missions

Before scoring, an evaluator-only preflight checked connected floor and a
reachable, readable 1.1 m witness view in a south-door service suite. The
doorway and sightlines differ from the previously opened north/central-door
layouts; ID7 was clear at its old site, moved north-east, moved with ID8 old,
or absent with ID8 old. Witness poses and true scene geometry stayed private.
In the moved camera-belief trace, the old view's ~45° capture projected ID7
behind the partition but exposed **zero visible gauge pixels** according to
evaluator-only render metadata; this is an occlusion control, not a hidden
policy cue.
[Feasibility](../../results/fresh_inspection_aware_20261028/evaluator_preflight/feasibility.json)
· [v2 pre-outcome freeze](../../results/fresh_inspection_aware_20261028/scenario_freeze_v2.json).
The v2 freeze records a launcher-only separation that restored the exact hash
of conventional `run.py` before any fresh outcome. No policy, case, budget or
template changed. All 140 method and 72 template hashes matched after scoring.

| Case | Selector | Complete physical task | Total sim s / m | Inspection m | First ID7 after handoff, m | Captures |
|---|---|---:|---:|---:|---:|---:|
| Clear old site | Nearest | Yes | 129 / 10.47 | 4.72 | 4.35 | 3 |
| Clear old site | Map visibility | Yes | 143 / 13.91 | 6.30 | 5.95 | 3 |
| Clear old site | Camera belief | Yes | 181 / 13.93 | 3.91 | 3.56 | 3 |
| Moved north-east | Nearest | Yes | 242 / 17.99 | 8.93 | 5.32 | 12 |
| Moved north-east | Map visibility | Yes | 214 / 17.04 | 10.64 | 7.85 | 12 |
| Moved north-east | Camera belief | Yes | 274 / 19.90 | 9.20 | 5.64 | 12 |
| ID8 old, ID7 moved | Nearest | Yes | 231 / 18.90 | 10.29 | 6.69 | 12 |
| ID8 old, ID7 moved | Map visibility | Yes | 248 / 21.57 | 12.07 | 9.17 | 13 |
| ID8 old, ID7 moved | Camera belief | Yes | 230 / 18.62 | 9.12 | 5.56 | 12 |
| ID7 absent, ID8 old | Nearest | Withheld | 327 / 17.38 | 7.72 | — | 32 |
| ID7 absent, ID8 old | Map visibility | Withheld | 393 / 24.54 | 14.53 | — | 32 |
| ID7 absent, ID8 old | Camera belief | Withheld | 337 / 17.73 | 8.64 | — | 32 |

All nine target-present final claimed captures passed the **independent physical
judge**, with correct ID7, required physical pose/heading/dwell and accepted
synthetic reading. The three absent missions made no completion claim; their
reason was `search_budget_exhausted`, **not proven absence**. Across twelve
attempts there were zero false completions, wrong-ID acceptances, contacts,
resets, interventions, infrastructure failures, or handoff failures. Minimum
clearance was 0.282 m; maximum handoff-seed position error was 0.151 m. All
twelve acquired maps had distinct hashes. This is **one environment, four
conditions, one run per selector/condition**, not a reliability estimate or
twelve independent sites. [Per-mission analysis](../../results/fresh_inspection_aware_20261028/analysis.json)
· [attempt ledger](../../results/fresh_inspection_aware_20261028/attempts.jsonl)
· [execution order](../../results/fresh_inspection_aware_20261028/execution.jsonl).

## Mechanism and limits

The fresh moved and wrong-asset runs show a camera-geometry distinction:
map visibility favored east; nearest and camera belief chose center. East
first decoded ID7 from ~3.0 m at ~71° to the asset face, yielding a projected
gauge face of 42–49 px. Center saw ID7 from ~3.8 m at ~49°, yielding ~63 px.
Both are below the renderer's 80 px usable-image threshold. **Identity was
available, inspection was not**; every successful policy then navigated to a
new physical view and read the gauge. The opened visibility result remains an
*earlier detection-station* effect, not evidence of a more readable first
view. [Actual ID8 old-view image](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/camera/observation_000.png)
→ [ID7 central detection image](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/camera/observation_009.png)
→ [final inspection image](../../results/fresh_inspection_aware_20261028/mission_wrong_asset_aware/inspection/camera/observation_011.png).

On the fresh layout, nearest and camera belief selected the **same center
station** for both moved conditions. Their modest cost differences cannot be
assigned to viewpoint logic: each independently remapped and acquired a
different handoff pose. Acquisition alone ranged from 75–131 s. Map
visibility's farther east station did not buy a completion advantage and cost
more inspection travel on those conditions, while it remained best on the
opened matched wrong-asset checkpoint. The camera-belief budget gate did not
reduce the selected station's expected detection mass in any fresh case; no
budget-aware benefit was tested by these outcomes. Selector computation was
~0.20 s median for camera belief and ~0.28 s for map visibility over logged
choices; nearest path computation was not separately instrumented.

The renderer, unique attached marker, known initial dock, commissioned old
view, 2-D map, and synthetic gauge imagery remain strong idealizations. The
actual first ID7 image, final reader result, and physical judge were separate
outputs. No policy used `private/scene.json`, evaluator object IDs, true pose,
or witness view; only the simulator camera and judge used private files.

## Decision and reproduction

**Keep the conventional pipeline.** This comparison does not demonstrate a
completion advantage for camera-belief search or a residual algorithmic gap.
Nearest sufficed on the new layout; map visibility was decisively better on
one opened matched case. The new arm exposed useful camera-angle diagnostics,
but its 0.75 detection likelihood and 2-D occlusion model are not calibrated.
The next justified research task is to measure detection and pose success
against real camera observations with known target presence, distance and
view angle, including actual obscuration, before claiming a probabilistic
viewpoint planner improves physical inspection. A stronger planning comparison
such as Taioli et al.'s would be warranted only if that measured observation
model exposes a consequential search failure under a real budget. No product
demand or physical camera transfer is established here. The integrated
mapping→handoff→search→verified-inspection capability remains usable evidence
of robotics systems work and a basis for a future physical pilot.

From the repository root, inspect retained results without simulation:

The historical analyzers below require the full local results archive. This
checkpoint retains their summaries and selected traces; see the
[checkpoint contents](checkpoint.md).

```bash
/usr/bin/python3 projects/site_localization/research/inspection_aware/analyze_matched.py
/usr/bin/python3 projects/site_localization/research/inspection_aware/analyze_fresh.py
```

For a new **uninterrupted** camera-belief reproduction on the archived fresh
wrong-asset template (choose a new output path; do not overwrite evidence):

```bash
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 projects/site_localization/research/inspection_aware/run_continuous.py \
  --template projects/site_localization/results/fresh_inspection_aware_20261028/templates/wrong_asset_aware \
  --out projects/site_localization/results/inspection_aware_repro_01 \
  --slam-params projects/site_localization/research/constraint_repair/slam_params_radius_1p5.yaml \
  --inspector-script projects/site_localization/research/inspection_aware/inspection.py \
  --seed 7601 --wall-limit 800
```

For a checkpoint diagnostic rerun, use `run_matched.py --single wrong_asset
aware --suffix _repro02` after sourcing ROS; it creates a simulator reset and
must never be counted as continuous-mission evidence. See the
[learner exercise](tutor_exercise.md) for a concrete camera-to-action trace.
