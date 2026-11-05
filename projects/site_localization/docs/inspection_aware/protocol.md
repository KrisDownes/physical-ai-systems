# Camera-aware viewpoint comparison — locked development plan

Recorded before the new policy is run on a mission. This is an engineering
comparison, not a claim of a new search algorithm or field reliability.

## Prior result and comparator choice

The opened [matched result](../post_repair_connected/matched_result.md) shows
that 2-D map visibility visited an eastern **detection** station earlier; its
first gauge image was still unreadable. The [fresh result](../post_repair_connected/fresh_result.md)
shows no full-mission completion advantage. We therefore score actual marker
identification and final inspection separately from geometric visibility.

[Yu and Chen (2024)](https://arxiv.org/html/2404.16507) use a camera frustum,
semantic voxels, RGB-D segmentation, 3-D TSDF, and two-stage target-surface
reconstruction. We inspected the full paper. Its *idea* of task-aware view gain
is relevant, but its inputs and reconstruction outcome do not fit our planar
marker-and-gauge task. The implementable conventional comparator follows
[Ye and Tsotsos (1999)](https://research.ibm.com/publications/sensor-planning-for-3d-object-search):
a discrete target-location belief, a recognition likelihood for each camera
view, negative-image updates, and detection probability versus travel cost.
Our 2-D finite-grid/heading implementation and completion-budget gate are
engineering adaptations, not a reproduction of their pan/tilt/zoom robot.

## Fixed information and model

All arms receive the same acquired occupancy map, known initial dock, old
commissioned asset pose, 3×3 candidate-station lattice, eight camera headings
per visited station, 1920×1080 images, 60° horizontal FOV, 16 cm attached
AprilTag ID7, AMCL estimate, frozen detector/reader, approach controller and
physical judge. No method reads scene, target truth, camera renderer metadata,
or evaluator witness routes. The new policy initializes equal mass over every
fourth collision-clear acquired-map cell and four face orientations. It does
**not** know the relocated position or orientation. Its image model uses map
ray clearance, ±25° bearing (a 5° FOV margin), front-facing cosine ≥0.35,
projected marker edge ≥16 px, 0.6–5 m range, and an explicitly uncalibrated
0.75 probability of decoding a geometrically eligible marker. Actual negative
ID7 images update the belief with Bayes' rule; they do not prove absence.
Candidate station utility is expected correct-ID detection mass with enough
estimated travel remaining for a hypothesized 1.1 m stand-off, divided by
planned travel +1 m. The 1.4 m margin covers stand-off motion/replanning;
it is fixed before missions. This policy predicts **detectability**, not gauge
readability. Actual reader acceptance, physical range 0.9–1.3 m, heading,
dwell, contact and complete mission judge remain independent and unchanged.
The 2-D ray cannot model elevated occlusion or camera texture; these are
specific possible model mismatches to inspect.

Arms: nearest; frozen 360° map-visible area per travel; camera-belief search.
Only station selection and new arm's belief update/budget admission differ.
The old-location first visit, rotation/capture sequence, navigation, marker
pose solver, inspection approach, one recovery action, and limits are shared.
All arms have 18 m/420 s **inspection** and 35 m whole-task travel allowances,
four search stations and 36 captures. Acquisition effort is counted separately.

## Questions and predictions

Primary: complete correct physical inspection of ID7 within budget, including
valid synthetic reading, safe range/heading/dwell and no contact. Secondary:
false completion, wrong ID, first correct ID detection time/distance and
image geometry, post-detection approach travel, total time/distance, captures,
contacts, abstention reason. Unit = mission; frames are correlated.

If camera-aware coverage is decisive, it should seek actual ID7 view earlier
than map-area search when the latter values irrelevant free space. If budget
admission is decisive, it should avoid a station from which the hypothesized
post-detection approach is infeasible; an early honest stop is not completion.
If the 2-D camera model is wrong, high predicted detection mass may yield no
ID7 pixels because of vertical occlusion, orientation, or renderer details.
If all arms complete with similar cost, the tested conditions do not establish
an improvement. A clear old-location control should need no new station.

### Development diagnostic (opened cases)

Use the archived acquired-map handoff checkpoints for `wrong_asset`, `moved`,
and `absent`; run the new arm once each. The previous nearest/visibility arms
are archived matched outcomes, with identical checkpoint files and budgets.
These restarts isolate search but **are not uninterrupted missions**.
Stop development after the three new-arm attempts. A missing ROS/Gazebo start
may be rerun with its original attempt retained; an algorithmic failure is
retained, not replaced. No tuning after these runs. A serious implementation
defect may be corrected once with revision and original trace preserved, then
the fresh cohort must use that corrected frozen revision.

### Fresh uninterrupted evaluation

Before outcomes, construct a connected service suite with the east-room door
near the south wall rather than the previous central/northern openings, and a
west screen. Predeclare ID7 old/clear, ID7 moved north-east, ID8 old with ID7
north-east, and ID7 absent with ID8 old. This changes sightlines while keeping
a feasible stand-off in the east room. Evaluator-only preflight checks connected
floor and a visible/readable witness image; witness coordinates never enter a
policy. Do not select cases based on a policy's result. Run all 12 complete
missions (4 cases × 3 arms), one attempt per cell, sequentially, seed 7601.
Stop on an infrastructure failure to diagnose it; keep the failed attempt and
explain any rerun. Do not retune after opening a fresh outcome. A missed target
is reported as budget exhaustion, never proof of absence.

No statistical reliability claim is possible from one environment and four
cases. Ground-truth geometry and object labels are evaluator-only; physical
outcome is judged from actual image records and independent simulator state.
