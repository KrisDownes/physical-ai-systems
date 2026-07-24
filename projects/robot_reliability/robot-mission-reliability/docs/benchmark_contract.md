# Benchmark Contract

## System Under Test

This benchmark evaluates the mission behavior of a simulated 2D differential-drive mobile robot. The robot begins at a fixed initial pose and attempts to reach three sequential waypoints.

Two software versions are compared. Both versions use the same point controller, which receives the estimated robot pose and current waypoint and produces commanded left- and right-wheel speeds. The versions differ in how they estimate the robot pose:

- `v1` uses command-based wheel odometry without GPS correction.
- `v2` uses the same odometry prediction and additionally applies GPS-based position and yaw corrections.

The simulation environment generates noisy GPS measurements and applies scenario-controlled left-wheel traction faults. Both versions generate and log GPS observations, but only `v2` uses them to correct its estimated pose.

Every execution produces a replayable event log containing:

- True and estimated pose
- Commanded wheel speeds
- GPS observations
- Localization residuals and ground-truth errors
- Mission events
- Fault-injection events
- Software-health events

Software-health latency is recorded as synthetic telemetry but does not delay commands, alter the simulation time step, or otherwise affect robot motion.

## Experimental Design

The benchmark varies three factors:

- Software version: `v1` or `v2`
- Scenario: `nominal`, `mild_slip`, `severe_slip`, or `noisy_gps`
- Deterministic seed: `0`, `1`, `2`, `3`, or `4`

The four scenarios and five seeds produce 20 matched scenario-seed conditions. Each condition is executed once with `v1` and once with `v2`.

The complete benchmark therefore contains:

- 20 `v1` executions
- 20 `v2` executions
- 20 matched version pairs
- 40 individual executions
- One independent event log and one result row per execution

### Software Versions

| Version | Pose-estimation behavior |
|---|---|
| `v1` | Integrates commanded wheel speeds as odometry without correcting the estimated pose from GPS. |
| `v2` | Uses the same odometry prediction, applies GPS position correction with gain `0.55`, and applies GPS-course yaw correction with gain `0.45` when successive GPS observations are separated by at least `0.18 m`. |

The simulator emits version-dependent software-health latency with an approximate base of `11 ms` for `v1` and `8 ms` for `v2`. These values are generated telemetry only and will not be interpreted as measured control-loop performance.

### Scenarios

| Scenario | Per-axis GPS noise standard deviation | Slip start | Slip end | Left-wheel traction | Purpose |
|---|---:|---:|---:|---:|---|
| `nominal` | `0.06 m` | `100.0 s` | `101.9 s` | `1.0` | Control condition with standard GPS noise and no active wheel-slip fault during the mission. |
| `mild_slip` | `0.06 m` | `5.0 s` | `9.0 s` | `0.65` | Tests recovery when the left wheel realizes 65% of its commanded speed during the slip interval. |
| `severe_slip` | `0.06 m` | `5.0 s` | `9.0 s` | `0.10` | Tests recovery when the left wheel realizes 10% of its commanded speed during the slip interval. |
| `noisy_gps` | `0.12 m` | `100.0 s` | `101.9 s` | `1.0` | Tests sensitivity to doubled GPS noise without an active wheel-slip fault. |

A traction value of `1.0` means the commanded left-wheel speed is fully realized. During an active slip fault, the commanded left-wheel speed is multiplied by the configured traction value.

The slip interval begins at `slip_start_s` and remains active while:

\[
\text{slip\_start\_s} \le t < \text{slip\_end\_s}
\]

### Deterministic Seeds and Pairing

The benchmark uses seeds:

```text
0, 1, 2, 3, 4
```

### Hypotheses

For the proposed experiment I estimate the following for each scenario across each seed :

- **Nominal**
    - v1 will reach all waypoints in the time window of 38s because there is no slip and the actual wheel speeds equal the commanded wheel speeds, so v1's odometry model matches the simulated motion.
    - v2 will reach all waypoints but slower than v1 due to the GPS noisy measurements affecting the wheel commands

- **Mild Slip**
    - v1 will not recover from the slip and again will not make it to all waypoints in the time window of 38s.
    - v2 will recover from the mild slip and reach all waypoints

- **Severe Slip**
    - v1 will not recover from the severe slip and will not reach all waypoints.
    - v2 will not recover from the severe slip and will not reach all waypoints, it will reach at least 1 waypoint. The GPS correction will not be enough to succeed. 

- **Noisy GPS**
    - v1 will perform the same as the Nominal scenario as it does not consume the GPS measurements, so it will reach all waypoints. 
    - v2 will suffer from the noisy GPS measurements and will not reach all waypoints in the time window of 38s. 


### Metrics

1. Mission Outcome
    - Mission success: Did the robot reach all three waypoints before the max time (38s)
    - waypoints reached: How many did it reach?
2. Completion Time
    - Time when the final waypoint was reached. Failed runtimes do not count towards the completion time.
3. Localization Error
    - The simulator records both true and estimated pose, we measure how wrong the robot believes it's location is using
    - position_error =  e_t = sqrt((x_true - x_estimated)^2 + (y_true - y_estimated)^2)
    - Localization RMSE: typical error over the entire run, with larger errors penalized more
        - sqrt(sum(e_t^2)/N)
    - Maximum localization error: worst error during run
4. Path Quality
    - Path Length: total distance actually travelled
    - Path efficiency: For successful missions, path efficiency is the straight line displacement from the initial pose to the final true position 
    divided by total path length. Failed missions have no path efficiency value.
    - Cross-track error: how far the true robot position moves sideways from the intended route
    - Maximum cross-track error: worst sideways deviation
    - For this experiment run cross-track error is essentially abs(y_true)

### Success Criteria
1. A mission success is reaching all waypoints before 38s
2. A benchmark success will be all 40 runs finish, produce 40 logs and result rows, and can be reproduced from the same configurations and seeds.
3. Each prediction is evaluated individually. Some may be supported while others are contradicted. 

### Non-Goals
- Performance on a physical robot
- Production ready localization, control, or safety
- Performance on routes and envs outside this simulation
- Actual motion smoothness, despite measuring path deviation
- Real control loop latency