from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path

from .replay import replay_events


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    software_version: str
    mission_success: bool
    waypoints_reached: int
    duration_s: float
    telemetry_samples: int
    localization_rmse_m: float
    localization_max_error_m: float
    detector_alerts: int
    detector_false_positives: int
    detection_latency_s: float | None
    log_sequence_gaps: int


def evaluate_run(path: Path, threshold_m: float = 0.20, consecutive_samples: int = 2) -> RunMetrics:
    events = list(replay_events(path))
    if not events:
        raise ValueError("run contains no events")

    telemetry = [event for event in events if event["event_type"] == "robot.telemetry"]
    mission = [event for event in events if event["event_type"] == "mission.event"]
    faults = [event for event in events if event["event_type"] == "fault.injection"]
    errors = [event["payload"]["localization"]["pose_error_m"] for event in telemetry]

    fault_start = next(
        (
            event["event_time_s"]
            for event in faults
            if event["payload"].get("name") == "left_wheel_slip" and event["payload"].get("active")
        ),
        None,
    )
    fault_end = next(
        (
            event["event_time_s"]
            for event in faults
            if event["payload"].get("name") == "left_wheel_slip" and not event["payload"].get("active")
        ),
        None,
    )

    alert_times: list[float] = []
    over_threshold = 0
    latched = False
    for event in telemetry:
        # The online detector may use only information available to the robot.
        # Simulation truth is reserved for the offline evaluator below.
        innovation = event["payload"]["localization"]["gps_innovation_m"]
        if innovation is None:
            continue
        if innovation >= threshold_m:
            over_threshold += 1
        else:
            over_threshold = 0
            latched = False
        if over_threshold >= consecutive_samples and not latched:
            alert_times.append(event["event_time_s"])
            latched = True

    # Residual effects can remain after traction returns, so alerts after the
    # injected fault are part of the same incident. Pre-fault alerts are false.
    false_positives = sum(
        1 for alert_time in alert_times if fault_start is None or alert_time < fault_start
    )
    first_valid_alert = next(
        (
            alert_time
            for alert_time in alert_times
            if fault_start is not None and alert_time >= fault_start
        ),
        None,
    )
    detection_latency = (
        round(first_valid_alert - fault_start, 6)
        if first_valid_alert is not None and fault_start is not None
        else None
    )

    terminal = mission[-1]["payload"]
    sequences = sorted(event["sequence"] for event in events)
    gaps = sum(max(0, current - previous - 1) for previous, current in zip(sequences, sequences[1:]))

    return RunMetrics(
        run_id=events[0]["run_id"],
        software_version=events[0]["software_version"],
        mission_success=terminal["name"] == "mission_completed",
        waypoints_reached=terminal["waypoints_reached"],
        duration_s=events[-1]["event_time_s"],
        telemetry_samples=len(telemetry),
        localization_rmse_m=sqrt(sum(error * error for error in errors) / len(errors)),
        localization_max_error_m=max(errors),
        detector_alerts=len(alert_times),
        detector_false_positives=false_positives,
        detection_latency_s=detection_latency,
        log_sequence_gaps=gaps,
    )


def write_report(metrics: list[RunMetrics], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(item) for item in metrics], indent=2) + "\n", encoding="utf-8")


def comparison_markdown(metrics: list[RunMetrics]) -> str:
    rows = [
        "| Version | Success | Waypoints | Duration (s) | Loc. RMSE (m) | Max error (m) | Detect latency (s) | False positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        latency = "n/a" if item.detection_latency_s is None else f"{item.detection_latency_s:.2f}"
        rows.append(
            f"| {item.software_version} | {str(item.mission_success).lower()} | "
            f"{item.waypoints_reached} | {item.duration_s:.1f} | "
            f"{item.localization_rmse_m:.3f} | {item.localization_max_error_m:.3f} | "
            f"{latency} | {item.detector_false_positives} |"
        )
    return "\n".join(rows) + "\n"
