# Connected-layout false association: conventional correction (2026-10-26)

## Decision

**A — a conventional configuration remedy is sufficient in this tested scope.**
The connected-layout transfer exposed a false cross-room registration/loop
association, not a remapping-to-inspection frame convention or an AMCL-only
handoff defect. Narrowing SLAM Toolbox's loop-candidate distance from 3.0 to
1.5 m, while leaving loop closing on, removed the recorded alias in five
same-scan replays. Two fresh, complete central-doorway missions then succeeded
without reset: a visible control and a moved ID7 with ID8 at the old site.
This is an engineering improvement, not an original algorithm or a reliability
estimate. The map may still need distant legitimate closures in larger sites.

## Causal evidence and limits

The frozen visible transfer's first gross mapper error is at 131.0 s: it moves
from 0.066 m physical position error at 130.3 s to 3.594 m. Its TF moves
3.655 m in 0.034 s while the physical robot moves 0.003 m. The first
independently wrong stored edge is **91→314**, linking scans at 36.0 and
131.0 s. Its relative translation `(-0.087, 1.166)` m disagrees with the
physical relative translation `(-0.487, -2.518)` m by 3.706 m. The first
node stays at its dock-relative origin. The pre-jump current estimate is
2.594 m from scan 91, outside the 1.5 m near-link gate but inside the 3 m
loop-candidate gate. The 8 m correlation search grid can contain a 3.7 m
alternative placement. [Raw edge and nodes](../../results/connected_layout_20261025/analysis/visible_graph_edges.csv),
[pose/TF truth comparison](../../results/connected_layout_20261025/analysis/causal_diagnostics.json),
[original scan and wheel rows](../../results/connected_layout_20261025/mission_visible/acquisition/mapper/sensor_frames.txt),
and [scan placement figure](../../results/constraint_repair_20261026/visible_scan_alias.png)
show the geometry. The final graph displaced scan 313 by 3.622 m from its
online pose immediately before the event, demonstrating retrospective map
deformation as well as an erroneous current pose.

The original serialized graph lacks a branch tag and matcher responses. Thus
the exact score of **91→314** cannot be recovered from that artifact. A
standalone replay of its serialized scans and wheel poses reproduced a false
**`TryCloseLoop`** event at nearby scan **93→311**, at 129.3 s (a 1.7 s shift
because the standalone replay does not run the live Ceres backend). Its coarse
response was **0.496 > 0.35**, coarse variances **0.0020 and 8.740 < 9**,
and fine response **0.684 > 0.45**; the installed setter squares the configured
coarse variance value 3.0. The graph gained sequential **310→311** and false
**93→311** edges at that node; the estimated x changed from 3.300 m at
scan 310 to -0.478 m at scan 311, while the recorded wheel motion remained
continuous. See [replay log](../../results/constraint_repair_20261026/visible_old_replay.log)
and [replay edges](../../results/constraint_repair_20261026/visible_old_replay_edges.csv).
This directly establishes a false loop in the replay; the archived 91→314
edge has the same geometry and timing, making the same mechanism strongly
supported but not branch-tag proven. The saved original matcher score must
not be presented as the replay score.

Deleting only original 91→314 from the final graph leaves 3.64 m endpoint
error: subsequent wrong links retain the distorted solution. A read-only
SE(2) weighted least-squares reoptimization dropping **all** links labeled
wrong using simulator truth reduced endpoint error from 3.65 to 0.107 m
(visible), 3.65 to 0.119 m (moved), 3.61 to 0.226 m (wrong asset), and 3.64
to 0.185 m (absent). The successful occluded graph remained at 0.034 m.
This is an **oracle diagnostic**, with diagonal recorded covariances and a
different offline optimizer; neither edge selection nor corrected poses enter
the robot. [Code](../../research/constraint_repair/reoptimize.py),
[visible single-edge counterfactual](../../results/constraint_repair_20261026/visible_drop_first.json),
and [all-edge oracle results](../../results/constraint_repair_20261026/visible_drop_all_oracle.json)
make the distinction inspectable. Huber and Cauchy fits to the already
corrupted saved graph stayed near 3.6 m error in the four failed cases; this
does not rule out stronger robust backend formulations or different online
initialization. The previous occluded acquisition had no link over the 0.5 m
evaluator diagnostic threshold and no large pose jump. These observations
contradict a universal transform sign, timestamp, or handoff-only explanation.

## Deployable remedy and recorded-scan stage

SLAM Toolbox's [installed Karto mapper logic](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/lib/karto_sdk/src/Mapper.cpp)
uses the candidate radius to choose possible loop chains, then coarse/fine
score gates before adding a link and optimizing. Its
[Ceres adapter](https://github.com/SteveMacenski/slam_toolbox/blob/2.8.5/solvers/ceres_solver.cpp)
offers Huber/Cauchy losses, but neither is a candidate-verification rule.
The selected correction is [one parameter line](../../research/constraint_repair/slam_params_radius_1p5.yaml):
`loop_search_maximum_distance: 1.5`; `do_loop_closing: true` and all other
parameters are unchanged. This is a geometric prior on plausible revisit
distance, not a truth-derived edge filter. It can reject useful closures when
odometry drift or revisits exceed the radius.

| Open recording | Original 3 m replay: final pose error / wrong cross-time links | 1.5 m replay: final pose error / wrong links | Legitimate cross-time links, old → narrow |
|---|---:|---:|---:|
| visible | 3.709 m / 12 | 0.160 m / 0 | 11 → 7 |
| moved | 3.713 m / 2 | 0.052 m / 0 | 7 → 7 |
| wrong asset | 3.651 m / 22 | 0.132 m / 0 | 2 → 4 |
| absent | 3.692 m / 4 | 0.116 m / 0 | 7 → 7 |
| previously successful occluded | 0.041 m / 0 | 0.041 m / 0 | 17 → 17 |

The replay reprocesses **identical serialized accepted scans and recorded
odometry**, but lacks the live Ceres backend and changed closed-loop sensing.
Counts of incorrect/legitimate links use interpolated simulator truth only in
the independent analysis (`>0.5 m` relative translation error is an
evaluator diagnostic, never a runtime threshold). Legitimate-link loss in
visible is a real tradeoff; the method does not just reject every loop.
[All replay metrics](../../results/constraint_repair_20261026/replay_comparison.json),
[replay implementation](../../research/constraint_repair/replay_karto.cpp), and
[evaluator](../../research/constraint_repair/analyze_replays.py) contain raw
and derived evidence. The original 3 m replay reproduces the alias and the
1.5 m replay does not; no parameter sweep or new anchor was used.

[Switchable constraints](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf)
and [dynamic covariance scaling](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/agarwal13icraws.pdf)
are established graph-backend approaches that require labeling loop factors
for robust weighting; the installed adapter does not directly expose them.
[Pairwise consistency maximization](https://robots.engin.umich.edu/publications/jmangelson-2018a.pdf)
uses multiple candidate constraints and path consistency, adding a candidate
management and uncertainty burden. Those alternatives are appropriate if a
tighter radius loses needed legitimate closure, not evidence of a new method
here. A robust loss alone need not reject a mutually reinforcing cluster of
confident but wrong matches, which is consistent with this offline result.

## Frozen fresh closed-loop result

The [pre-execution protocol](protocol.md) and [fresh freeze](../../results/constraint_repair_20261026/fresh/freeze.json)
precede execution. The central-doorway layout differs in connectivity and
sightlines from the opened north-doorway layout; it retains the same 6.4 m
enclosure and physical task. [Evaluator preflight](../../results/constraint_repair_20261026/fresh/evaluator_preflight/feasibility.json)
confirmed connected reachable floor and a feasible, readable target witness;
the witness/path are **not method inputs**. Both cases used the same known
initial dock, acquisition/exploration, map/estimated-pose handoff, frozen
marker and inspection policy, physical criteria, and budgets. No map, robot,
reader or completion-rule tuning occurred after outcomes.

| Fresh case | Complete physical mission | Truth-scored map endpoint / max error | Acquisition / whole sim time | Whole distance | Observations | Clearance / contacts |
|---|---|---:|---:|---:|---:|---:|
| ID7 old location, visible | Yes | 0.088 / 0.088 m | 102.1 / 148.8 s | 13.56 m | 3 | 0.296 m / 0 |
| ID8 old, ID7 moved east | Yes | 0.005 / 0.028 m | 94.1 / 213.4 s | 17.60 m | 11 | 0.336 m / 0 |

Both used one simulator lifetime, one initial dock alignment, zero resets,
zero interventions, zero false completions, and no collision. The moved case
actually observed ID8 at 137.156 s and withheld target completion; it
observed ID7 at 182.070 s and 182.818 s, then completed on a fresh valid
observation at 210.426 s. [Moved raw mission](../../results/constraint_repair_20261026/fresh/mission_fresh_moved_wrong/outcome.json)
and [method decision trace](../../results/constraint_repair_20261026/fresh/mission_fresh_moved_wrong/inspection/method.jsonl)
show this distinction. Its final reading was 6.204 MPa; the independent
judge measured the position/view/dwell contract separately. The visible
case's final view was 1.125 m away, 2.70° off axis, with 2.312 s stationary
dwell. The moved case's final view was 1.090 m away, 1.58° off axis, with
2.312 s dwell. Both recorded a valid ID7/6.204 MPa reading. The acquired live graphs had zero
truth-identified wrong cross-time links (27 and 20 legitimate links under the
same diagnostic). [All fresh mission metrics](../../results/constraint_repair_20261026/fresh/analysis/missions.json)
and [decoded graph CSVs](../../results/constraint_repair_20261026/fresh/analysis/fresh_moved_wrong_graph_edges.csv)
are reproducible.

**All attempted executions are retained.** The first fresh-visible attempt
failed startup before Gazebo launched because sandboxed ROS could not create
DDS UDP sockets. It produced no interpretable robot outcome and remains
[recorded](../../results/constraint_repair_20261026/fresh/mission_fresh_visible/run.json).
The same frozen case was rerun once in a new directory with ROS socket
permissions; the moved case ran once. [Attempt ledger](../../results/constraint_repair_20261026/fresh/attempts.jsonl)
has three starts and three finishes; all owned processes were reaped. Source,
template and evaluator hashes still match the fresh freeze. The
[raw successful visible mission](../../results/constraint_repair_20261026/fresh/mission_fresh_visible_rerun01/outcome.json)
is separate from its failed infrastructure attempt.

The older connected-layout execution freeze now detects one intentional
working-source change: `research/continuous_asset/run.py` gained an optional
`--slam-params` input so the radius correction could be supplied explicitly.
Its historical [source snapshot](../../results/connected_layout_20261025/mission_visible/source/run.py)
and all old outcomes remain intact. The new mission's `freeze.json` records
the exact updated runner and copied SLAM configuration. No older result was
rescored or overwritten.

## What this establishes and next decision

The practical mission problem is preserving intended-asset inspection after
remapping in a connected site. In these simulations, a conventional
candidate-radius correction restored useful map pose and enabled two complete
continuous missions, including a wrong-asset opportunity. It does **not**
establish field reliability, real-camera/gauge transfer, performance under
larger odometric drift, or a field-level research gap. The one new layout and
two interpretable missions are correlated evidence, not two independent
environments. The known start/dock, ideal planar sensors/calibration,
synthetic camera, and commissioned identity remain assistance.

Next bounded implementation task: keep this corrected conventional pipeline
fixed and test an independently designed longer connected layout where a
legitimate revisit is expected to be more than 1.5 m apart in the pose prior.
Record whether the narrow radius loses a closure needed for physical
inspection; compare against the original radius and, only if a meaningful
tradeoff appears, an established uncertainty/consistency-aware loop treatment.
Do not make a novelty claim from the current failure. The usefulness of the
complete autonomous search-to-inspection system remains even if that later
hypothesis fails.

## Reproduction and learner handoff

From the repository root, with the installed ROS Jazzy environment:

```bash
bash projects/site_localization/research/constraint_repair/build_replay.sh /tmp/site-replay-karto
LD_LIBRARY_PATH=/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-} /tmp/site-replay-karto \
  projects/site_localization/results/connected_layout_20261025/mission_visible/acquisition/mapper/session.data \
  3.0 /tmp/site-visible-3m-edges.csv > /tmp/site-visible-3m.log 2>&1
LD_LIBRARY_PATH=/opt/ros/jazzy/lib:${LD_LIBRARY_PATH:-} /tmp/site-replay-karto \
  projects/site_localization/results/connected_layout_20261025/mission_visible/acquisition/mapper/session.data \
  1.5 /tmp/site-visible-1p5m-edges.csv > /tmp/site-visible-1p5m.log 2>&1
/usr/bin/python3 projects/site_localization/research/constraint_repair/reoptimize.py \
  visible --drop 91:314 --out /tmp/site-visible-single-edge-oracle.json
/usr/bin/python3 projects/site_localization/research/constraint_repair/analyze_replays.py
/usr/bin/python3 projects/site_localization/research/constraint_repair/summarize_fresh.py
source /opt/ros/jazzy/setup.bash
source projects/site_localization/vendor/setup_amcl.bash
/usr/bin/python3 projects/site_localization/research/continuous_asset/run.py \
  --template projects/site_localization/results/constraint_repair_20261026/fresh/templates/fresh_moved_wrong \
  --out projects/site_localization/results/constraint_repair_20261026/fresh/new_development_repeat \
  --slam-params projects/site_localization/research/constraint_repair/slam_params_radius_1p5.yaml \
  --seed 7403 --wall-limit 800
```

The last command launches a **new development repeat**, not a reconstruction
of a frozen mission; leave earlier outputs untouched and do not run alongside
another simulator. The [learner exercise](tutor_exercise.md) asks for a
prediction and transform calculation from the failed 91→314 link. The
[solution](tutor_solution.md) is separate; tutor notes were not edited.
