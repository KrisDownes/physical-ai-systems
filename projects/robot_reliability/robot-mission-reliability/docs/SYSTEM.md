# System design and engineering reasoning

## Operational question

The rover has one mission: reach `(2, 0)`, `(4, 0)`, and `(6, 0)` in order within 38 seconds. A waypoint counts only when the physical pose is within 0.22 m. Reaching all three is success.

This definition comes before the implementation. Without it, “the robot moved” or “the pipeline processed data” could be mistaken for success.

## State and evidence boundaries

| Quantity | Available onboard? | Used by online detector? | Used by offline evaluator? |
|---|---:|---:|---:|
| Wheel-derived pose estimate | yes | yes | yes |
| Noisy global position | periodically | yes | yes |
| Global-position innovation | yes | yes | yes |
| Simulation truth pose | no | no | yes |
| Injected-fault label | no | no | yes |
| Mission outcome | yes | no | yes |

This separation prevents **ground-truth leakage**. An anomaly detector that compares its estimate directly with simulator truth can score well but cannot exist on a real robot.

## Failure mechanism

During the fault, the left wheel achieves only 28% of its intended ground speed. The encoder/odometry model integrates the commanded wheel motion and therefore remains confident that the rover is traveling straight. The true rover curves away.

The failure is useful because it crosses several layers:

- physical: traction changes;
- sensing: wheel rotation is no longer equivalent to ground displacement;
- estimation: odometry diverges from position measurements;
- autonomy: the controller acts on the wrong pose;
- mission: the rover can stall at a waypoint it falsely believes it reached;
- data: diagnosis requires synchronized commands, estimates, observations, fault timing, and mission state.

## Event contract

Every record has a common envelope:

| Field | Purpose |
|---|---|
| `event_id` | globally unique record identity |
| `run_id` | joins every plane from one experiment |
| `mission_id` | identifies the operational task |
| `robot_id` | identifies the producing asset |
| `software_version` | makes regression comparison possible |
| `event_type` | selects the payload contract |
| `sequence` | exposes duplicates and missing records |
| `event_time_s` | orders what happened on the robot timeline |
| `recorded_at` | records when the logger serialized the event |
| `schema_version` | supports controlled contract evolution |
| `payload` | event-specific evidence |

Wall-clock timestamps alone are not enough. Distributed robot systems experience buffering, retries, and delayed arrival; diagnosis needs event time plus per-run ordering.

## Detector and evaluator are different components

The online-style detector alerts after two consecutive global-position updates have an innovation of at least 0.20 m. It uses information a robot could possess.

The offline evaluator uses truth to calculate localization RMSE, maximum error, mission outcome, and detection latency relative to injected fault time. It judges the detector and controller; it is not deployed as the detector.

## Why JSONL first

At this stage the hard question is whether the events carry the right meaning and whether the run is replayable. JSONL makes every record inspectable and keeps the experiment deterministic. Introducing Kafka now would test deployment mechanics while leaving the core robotics semantics unchanged.

The transport upgrade is justified when the project needs concurrent producers, backpressure, delivery semantics, or live consumers. The data contract and evaluator should survive that upgrade unchanged.

## Version decision

A useful release gate could be:

- mission success rate at least 95% across the scenario matrix;
- localization RMSE below 0.20 m;
- no pre-fault alerts in nominal runs;
- p95 detection latency below 5 seconds;
- no event sequence gaps in a local run.

The current single-seed experiment demonstrates the measurement method, not those population-level claims.

## Running the Complete Benchmark

From the `robot-mission-reliability` project directory, with the Python environment active, run:

```bash
make benchmark
```
This command:

1. Executes all 40 scenario, software-version, and seed combinations.
2. Writes one replayable JSONL log per execution.
3. Writes the 40-row machine-readable CSV.
4. Aggregates results across the five deterministic seeds.
5. Generates the scenario comparison report.
6. Generates the mission-success and cross-track-error plots.
7. Generates the representative `severe_slip`, `v2`, seed `0` replay GIF.

Generated outputs:

- `artifacts/runs/`
- `artifacts/results.csv`
- `docs/scenario_comparison.md`
- `docs/assets/success_rate.svg`
- `docs/assets/cross_track_rmse.svg`
- `docs/assets/severe_slip_v2_seed_0.gif`

