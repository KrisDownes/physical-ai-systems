from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import atan2, hypot
from pathlib import Path
from uuid import uuid4

from .contracts import EventEnvelope
from .models import Pose2D, RobotConfig, integrate_differential_drive, point_controller, wrap_angle


@dataclass(frozen=True)
class RunConfig:
    software_version: str
    seed: int = 7
    max_time_s: float = 38.0
    gps_period_s: float = 0.5
    gps_noise_std_m: float = 0.06
    waypoint_tolerance_m: float = 0.22
    slip_start_s: float = 5.0
    slip_end_s: float = 9.0
    left_wheel_traction: float = 0.28


class EventWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")

    def write(self, event: EventEnvelope) -> None:
        self._file.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._file.close()


def _make_event(
    *,
    run_id: str,
    config: RunConfig,
    event_type: str,
    sequence: int,
    time_s: float,
    payload: dict,
) -> EventEnvelope:
    return EventEnvelope.create(
        run_id=run_id,
        mission_id="warehouse-route-001",
        robot_id="rover-001",
        software_version=config.software_version,
        event_type=event_type,
        sequence=sequence,
        event_time_s=time_s,
        payload=payload,
    )


def run_simulation(output_path: Path, config: RunConfig) -> str:
    """Run one deterministic mission and write an interleaved, replayable event log."""

    rng = random.Random(config.seed)
    robot = RobotConfig()
    truth = Pose2D()
    estimate = Pose2D()
    waypoints = [(2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]
    waypoint_index = 0
    time_s = 0.0
    sequence = 0
    next_gps_s = 0.0
    previous_gps: tuple[float, float] | None = None
    run_id = str(uuid4())
    writer = EventWriter(output_path)

    def emit(event_type: str, payload: dict) -> None:
        nonlocal sequence
        writer.write(
            _make_event(
                run_id=run_id,
                config=config,
                event_type=event_type,
                sequence=sequence,
                time_s=time_s,
                payload=payload,
            )
        )
        sequence += 1

    emit("mission.event", {"name": "mission_started", "waypoints": waypoints})
    fault_active_previous = False

    while time_s < config.max_time_s and waypoint_index < len(waypoints):
        target = waypoints[waypoint_index]
        commanded_left, commanded_right = point_controller(estimate, target, robot)
        fault_active = config.slip_start_s <= time_s < config.slip_end_s
        actual_left = commanded_left * (config.left_wheel_traction if fault_active else 1.0)
        actual_right = commanded_right

        if fault_active != fault_active_previous:
            emit(
                "fault.injection",
                {
                    "name": "left_wheel_slip",
                    "active": fault_active,
                    "traction": config.left_wheel_traction if fault_active else 1.0,
                },
            )
            fault_active_previous = fault_active

        truth = integrate_differential_drive(
            truth, actual_left, actual_right, robot.dt_s, robot.wheel_base_m
        )
        estimate = integrate_differential_drive(
            estimate, commanded_left, commanded_right, robot.dt_s, robot.wheel_base_m
        )

        gps_available = time_s + 1e-9 >= next_gps_s
        gps_x = gps_y = None
        innovation_m = None
        if gps_available:
            gps_x = truth.x_m + rng.gauss(0.0, config.gps_noise_std_m)
            gps_y = truth.y_m + rng.gauss(0.0, config.gps_noise_std_m)
            innovation_m = ((gps_x - estimate.x_m) ** 2 + (gps_y - estimate.y_m) ** 2) ** 0.5
            next_gps_s += config.gps_period_s
            if config.software_version == "v2":
                # A deliberately simple correction. The point is to evaluate a behavior change,
                # not to pretend this is a production localization stack.
                gain = 0.55
                estimate.x_m += gain * (gps_x - estimate.x_m)
                estimate.y_m += gain * (gps_y - estimate.y_m)
                if previous_gps is not None:
                    gps_dx = gps_x - previous_gps[0]
                    gps_dy = gps_y - previous_gps[1]
                    if hypot(gps_dx, gps_dy) >= 0.18:
                        gps_course = atan2(gps_dy, gps_dx)
                        yaw_gain = 0.45
                        estimate.yaw_rad = wrap_angle(
                            estimate.yaw_rad
                            + yaw_gain * wrap_angle(gps_course - estimate.yaw_rad)
                        )
            previous_gps = (gps_x, gps_y)

        pose_error_m = truth.distance_to(estimate.x_m, estimate.y_m)
        emit(
            "robot.telemetry",
            {
                "truth": {"x_m": truth.x_m, "y_m": truth.y_m, "yaw_rad": truth.yaw_rad},
                "estimate": {
                    "x_m": estimate.x_m,
                    "y_m": estimate.y_m,
                    "yaw_rad": estimate.yaw_rad,
                },
                "command": {"left_mps": commanded_left, "right_mps": commanded_right},
                "gps": {"available": gps_available, "x_m": gps_x, "y_m": gps_y},
                "localization": {"pose_error_m": pose_error_m, "gps_innovation_m": innovation_m},
                "mission": {"waypoint_index": waypoint_index, "target_x_m": target[0], "target_y_m": target[1]},
            },
        )

        if int(round(time_s / robot.dt_s)) % 10 == 0:
            base_latency = 11.0 if config.software_version == "v1" else 8.0
            emit(
                "software.health",
                {
                    "control_loop_latency_ms": max(0.1, rng.gauss(base_latency, 1.5)),
                    "queue_depth": 0,
                    "cpu_percent": max(0.0, rng.gauss(24.0, 3.0)),
                    "dropped_messages": 0,
                },
            )

        if truth.distance_to(*target) <= config.waypoint_tolerance_m:
            emit(
                "mission.event",
                {"name": "waypoint_reached", "waypoint_index": waypoint_index},
            )
            waypoint_index += 1

        time_s = round(time_s + robot.dt_s, 10)

    success = waypoint_index == len(waypoints)
    emit(
        "mission.event",
        {
            "name": "mission_completed" if success else "mission_failed",
            "reason": None if success else "timeout",
            "waypoints_reached": waypoint_index,
        },
    )
    writer.close()
    return run_id
