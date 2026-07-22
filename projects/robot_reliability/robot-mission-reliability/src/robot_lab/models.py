from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


@dataclass
class Pose2D:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    def distance_to(self, x_m: float, y_m: float) -> float:
        return hypot(x_m - self.x_m, y_m - self.y_m)


@dataclass(frozen=True)
class RobotConfig:
    wheel_base_m: float = 0.42
    max_linear_mps: float = 0.8
    max_angular_rps: float = 1.8
    dt_s: float = 0.1


def integrate_differential_drive(
    pose: Pose2D,
    left_mps: float,
    right_mps: float,
    dt_s: float,
    wheel_base_m: float,
) -> Pose2D:
    linear = 0.5 * (left_mps + right_mps)
    angular = (right_mps - left_mps) / wheel_base_m
    midpoint_yaw = pose.yaw_rad + 0.5 * angular * dt_s
    return Pose2D(
        x_m=pose.x_m + linear * cos(midpoint_yaw) * dt_s,
        y_m=pose.y_m + linear * sin(midpoint_yaw) * dt_s,
        yaw_rad=wrap_angle(pose.yaw_rad + angular * dt_s),
    )


def point_controller(
    pose: Pose2D,
    target: tuple[float, float],
    config: RobotConfig,
) -> tuple[float, float]:
    dx = target[0] - pose.x_m
    dy = target[1] - pose.y_m
    distance = hypot(dx, dy)
    heading_error = wrap_angle(atan2(dy, dx) - pose.yaw_rad)
    linear = min(config.max_linear_mps, 0.9 * distance)
    if abs(heading_error) > 0.7:
        linear *= 0.25
    angular = max(-config.max_angular_rps, min(config.max_angular_rps, 2.4 * heading_error))
    left = linear - 0.5 * config.wheel_base_m * angular
    right = linear + 0.5 * config.wheel_base_m * angular
    return left, right
