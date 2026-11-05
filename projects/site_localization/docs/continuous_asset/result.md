# Continuous acquired-map inspection — 2026-10-25

The robot now runs **sensor-driven exploration → map/estimated-pose handoff → autonomous search → intended-asset inspection in one uninterrupted simulation**. The frozen fresh runs completed all three target-present missions. The absent-target run reported bounded-search non-completion. No world/odometry reset, second dock initialization, operator intervention, contact or terminal navigation failure occurred in those four runs.

This removes the acquisition-to-inspection reset dependency. Known **initial** dock assistance remains. These are conventional integration results, not a new algorithm, field validation or reliability estimate.

## Contract and failure diagnosis

The earlier acquisition contract explicitly required dock return; both earlier failures remain failures. Return was a staging convenience for restarting inspection, not an established operational requirement. Frame alignment already uses the first stationary SLAM node and the initial commissioned dock. The new contract explicitly requires **neither intermediate nor final dock return**. It retains identity, reading, position, heading, dwell and contact requirements. See the [pre-execution protocol](protocol.md) and [information handoff](handoff.md).

Raw replay distinguishes the old failures:

- **Unchanged:** at53.518s the requested translating/turning arc leaves the saved planner safety region. The estimated start later falls outside inflated free space; dock planning fails at56.386s. This is a planning-start/clearance failure. About2.8–2.9cm localization error coexists with tracking/stopping/grid effects; saved maps cannot uniquely apportion their contributions. It was not budget exhaustion or an observed contact.
- **Moved:** the133.014s commitment required5.203m outbound plus4.731m return with6.051m left. The between-leg threshold admitted an already infeasible commitment. Later interruption delayed return; the20m cap terminated acquisition. Return-cost admission was an identified defect independent of that interruption.

[Diagnosis and primary implementation sources](diagnosis.md), [replay measurements](../../results/continuous_asset_20261025/failure_replay.json), and [connector replay](../../results/continuous_asset_20261025/connector_replay.json) retain the causal evidence and its limits.

The scoped correction reserves1.25×estimated travel plus stopping/replanning allowances; routes with0.52m inflation while enforcing the original0.45m safety region; and checks estimated command arcs before translation. The first corrected unchanged run **failed** because its start satisfied hard safety but not the extra routing preference. V2 adds a≤0.3m connector checked entirely against hard-safe free space. It does not clear unknown cells or relax physical criteria. The two subsequent old-contract reproductions passed independent dock checks at16.40m and14.50m. The combined corrections are supported; their separate closed-loop effects were not ablated. The arc guard modified no commands in the new runs, so replay detection alone is not proof of its independent physical benefit.

All four development executions are retained, including that failure. The continuous moved development mission then passed at303.01s/21.76m. [Development changes](../../results/continuous_asset_20261025/development_log.jsonl) and per-attempt source snapshots preserve revisions. Before that continuous run, handoff preparation also gained a serialization-response check, final stopped-mapper TF/wheel refresh, and a wait for two seconds of actual AMCL samples. Fresh physics began only after freezing the resulting pipeline.

## Fresh mission results

[Freeze](../../results/continuous_asset_20261025/final_freeze.json): SHA256 `0db51ef5d6f22f32dcdf08a85991c67addad29b2bae941d96ba12015895443c6`. Four acquisitions and four complete mission attempts, each starting once at the known initial dock; one fresh layout variation, two planar geometry conditions, no repeated seeds. This is not four independent sites. No retries or retuning followed fresh outcomes.

| Case | Complete physical inspection | Final decision | Acquisition s / physical m | Whole mission s / physical m | Images |
|---|---|---|---:|---:|---:|
| Unchanged target location | Yes | Fresh identity and inspection |130.19 /12.52|179.89 /16.69|3|
| Moved target + changed surroundings | Yes | Fresh identity and inspection |122.06 /10.23|291.21 /20.82|19|
| ID8 at old site, moved ID7 | Yes | Fresh ID7 and inspection |138.30 /12.91|318.24 /23.42|19|
| ID7 absent, ID8 present | **No** | Search budget exhausted; no absence proof |140.35 /12.49|406.54 /22.19|32|

Whole-mission time includes handoff and termination tail. Acquisition-to-inspection preparation consumes additional time beyond the acquisition decision; inspection-stage times were43.99,163.23,174.32 and260.60s, respectively, measured from the final handoff state. The [mission ledger](../../results/continuous_asset_20261025/analysis/mission_ledger.csv) records stage wheel distances, wall costs and every attempt separately.

All fresh runs had zero false-completion claims, wrong-asset acceptances, incorrect numeric acceptances, inspection-reader abstentions, recoveries and contacts; clearance minima were0.320,0.306,0.334 and0.324m. No terminal navigation failure occurred. One moved-case frontier was interrupted at24.014s by new-map clearance information and replaced; this is preserved as an acquisition goal failure, not hidden as flawless navigation. All four acquisitions stopped on the predeclared next-leg reserve, reporting about36.15–36.39m² known free area, **not certified environmental coverage**. The absent mission made32 observations at four search positions, observed ID8, never verified ID7 and claimed no exhaustive absence.

Final inspections satisfied the same physical contract:

| Present case | Range m | Heading error deg | Stationary dwell s | Reading MPa |
|---|---:|---:|---:|---:|
| Unchanged |1.129|1.38|2.346|6.204|
| Moved |1.076|1.10|2.312|6.204|
| Wrong asset opportunity |1.094|2.80|2.346|6.204|

Each claimed final observation was independently valid, not merely some earlier valid frame. The reference interval remains[6,6.5]MPa with0.5MPa tolerance; worst-case error within that interval was0.296MPa. Identical values reflect the same synthetic commissioned dial texture, not changing-pressure or real-camera validation. Frames/stops are correlated within missions; no deployment reliability interval is warranted from this one layout variation.

## Handoff and information audit

All four handoffs retained one server lifetime, monotonic truth sampling (maximum gap0.034s), unchanged wheel odometry and no physical motion during the stopped transfer. The first AMCL-ready estimates arrived3.57–3.66s after the final estimated-state timestamp. Independent seed errors were6.76,10.36,12.13 and12.71cm /0.25,1.28,1.35 and2.34deg. They are evaluator measurements, not policy inputs. [Per-stage errors, target projection errors, final evidence and guard diagnostics](../../results/continuous_asset_20261025/analysis/diagnostics.json) include the full range rather than only successful-frame summaries.

The policy receives live lidar/wheel odometry, the acquired occupancy map, SLAM/AMCL estimates, decoded camera pixels, commissioned marker/asset geometry and bounded region/search settings. It does not read changed-world geometry, relocated target poses, object IDs or truth. The sensor renderer uses geometric projections/occlusions and the simulated camera pose to produce images; the policy decodes pixels. Scenario/evaluator truth files remain accessible to the simulator processes, not imported into the method. This is audited code separation, not an operating-system security boundary. Camera acquisition is disabled during mapping, so no hidden target discoveries are transferred.

Remaining assumptions: known initial dock and first-node alignment; static planar scene during each run; configured lidar/camera extrinsics; unique rigidly attached ID7; commissioned geometry/reader; fixed search region and candidate lattice; idealized synthetic imaging. AMCL seed floors0.15m/10deg are engineering choices, not calibrated joint uncertainty. SLAM and AMCL share lidar/map error. Unknown space remains blocked; neither exploration termination nor four search stations certify observable coverage.

## Decision

The reset dependency is resolved within this scope. Conventional corrections were sufficient for the recorded return conditions and continuous integration; no residual failure here establishes an original research gap. Preserve this as the working baseline.

The next bounded robotics validation should keep this pipeline fixed in **one genuinely different connected layout with partial-map/occluded search coverage**, using one reachable moved target and one target-unobserved condition. The specific uncertainty is whether acquisition leaves reachable informative inspection viewpoints before its budget ends, and whether non-completion reasons match the actual visibility/access limitation. Record map-known reachability and observed visibility before consulting the evaluator. A failure would first require comparison with conventional frontier/object-search remedies, not an immediate novel acceptance rule. This is a proposed next task, not another experiment run here.

The current result provides inspectable ROS integration, frame transfer, planning, perception and independent evaluation evidence. Real same-asset camera transfer, dynamic obstruction, localization robustness, docking after inspection and product demand remain unvalidated. No user learning mastery is claimed; the [tutor exercise](tutor_exercise.md) uses the actual handoff.

[Reproduce](reproduce.md) · [Uninterrupted demonstration](../../results/continuous_asset_20261025/analysis/continuous_demo.mp4) · [Handoff trace](../../results/continuous_asset_20261025/analysis/handoff_trace.png) · [Raw attempts](../../results/continuous_asset_20261025/attempts.jsonl) · [Integrity audit](../../results/continuous_asset_20261025/analysis/integrity.json)
