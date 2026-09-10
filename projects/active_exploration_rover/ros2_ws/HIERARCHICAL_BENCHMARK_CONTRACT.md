# Hierarchical exploration contract v2 (2026-09-09, pre-model)

Responsibility: LLM chooses one eligible online-map approach ID; existing custom
ExplorationPolicy / grid planner, PathFollower and ObstacleGuard execute and
recover locally. This is not Nav2. Original direct-control benchmark is preserved.

```
Camera + lidar + SLAM/TF + shared candidate filter + recent goal outcomes
                         ↓ event snapshot
          ClassicalSelector | LLM destination selector
                         ↓ current approach ID, versioned
                  current-grid validation
                         ↓
          existing grid planner → path follower → obstacle guard
                         ↓
           independent steady-clock safety gate → cmd_vel
```

Both hierarchical selectors share build_reachable_candidates, filter_candidates,
weighted classical selector only when explicitly selected, BFS/escape corridor,
bounded same-target fresh approaches, path following, .15 m/s / .60 rad/s limits,
.45/.55 m front/rear safety hysteresis, SLAM and spawn (0,0,.02,0), kd_world.
LLM additionally sees camera/lidar/map rendering; classical scoring uses path
cost and component size. Both see the same eligible approach set and memory
exclusions. No raw representative is silently substituted for a selected approach.
The shared inherited escape-corridor and recovery behaviors are classical motion
responsibilities, not LLM decisions. Neither selector receives simulator truth.

Destination requests occur only when no retained valid goal exists: startup,
arrival or bounded progress/planning failure. At most one transport worker is
active; responses carry request/map/goal generations and are revalidated against
current eligible world points within .75 map-cell resolution. Superseded responses
are discarded. Ordinary map-cycle replanning does not request new decisions.
The conservative executive stops while a necessary decision is pending; it does
not attempt speculative motion. Cancellation/completion latch stopped forever.

Defaults (ROS parameters): original stuck window 6 sim s / progress .05 m /
alignment .3927 rad; 3 path failures, 3 same-target fresh approaches; guard blockage
4 sim s, reverse 1.5 sim s then turn 2.75 sim s; temporary blacklist 30 sim s.
Request debounce 1 wall s, response timeout 60 wall s, sensor receipt/header
limit 1 s, TF .5 sim s, map 5 sim s, clock-stall limit 1 wall s, resume requires
1 continuous safe wall s. Measured forward/yaw-rate velocity covariance ceiling .5 (m²/s² or rad²/s²).
Absolute EKF position covariance is not SLAM map uncertainty: velocity-only EKF
position variance is unbounded. Map→odom jumps >.25 m or 5 degrees stop locally;
finite/stale TF checks remain. Obsolete goals debounce for 2 sim seconds.
Safety gate runs at 20 Hz steady time and rejects stale guard commands >.3 wall s.
Rotated/non-map occupancy frames fail closed rather than misproject destinations.
Known residual: these velocity-covariance/TF checks do not certify localization
accuracy; no ground-truth localization signal enters control.

Completion and all hard gates remain exactly BENCHMARK_CONTRACT.md / original
mission evaluator. Known-grid means known rectangular-grid cells / all grid cells,
not accessible-area coverage. No contact sensor: interventions are not collisions.
Fresh simulation/SLAM and independent bags per method; GUI if available. Completion
cancels work/stops motion, followed by original post-completion recording.
600 episode wall seconds, no automatic episode retries. Mock diagnostic <=120
wall seconds, never reported as model performance.

Record executive.jsonl (reasons, goal generations, selections/arrivals/failures,
changed paths, recovery, state duration by cause), exact goal_observations.jsonl,
app-server sent/received monotonic timestamps and full external transcript, token
usage notifications when exposed. Replan count means changed published grid paths,
not planner calls. Waiting-for-selector, safety and controller-stall time remain
separate; simulator clock is never used for wall latency. State durations are
operational proxies: executing includes tracking/turning and guard blockage.
Mission memory exposes last 12 outcomes; full event history remains on disk;
persistent thread retains earlier turns and no manual history truncation occurs.

## Decision-turn budget (explicit user requirement revision)
Previous version: HIERARCHICAL_BENCHMARK_CONTRACT_v1.md, preserved unchanged.
At most **20 externally submitted LLM decision turns per episode**, enforced
at the transport turn/start write boundary. Failed submissions conservatively
consume a slot. No application-level automatic retries or model substitutions.
Episode wall limit is 600 seconds. Model remains gpt-5.6-luna / low in one
persistent session. Log exposed tokens, internal retry/compaction notifications
and response IDs wherever exposed; retain raw RPC events.
Internal inference requests may exceed submitted turns. This is **not** a
guaranteed token or monetary cap and does not enforce internal inference counts.
Model execution is disabled by default; a future authorized comparison explicitly
passes --enable-model. This implementation task permits mocks only.

Prediction: fewer externally submitted model decisions and less measured model-waiting time while
retaining original exploration completion. Smooth motion alone is not success.
