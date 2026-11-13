"""Agent-authored kinematic plant; angles and increments are radians.

The fixed-step response is kinematic, not a motor or rigid-body dynamics model.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlantConfig:
    link_lengths_m: tuple[float, float] = (0.30, 0.20)
    joint_min_rad: tuple[float, float] = (-np.pi, -np.pi)
    joint_max_rad: tuple[float, float] = (np.pi, np.pi)
    max_increment_rad: float = 0.1
    response_fraction: float = 0.8
    step_duration_s: float = 0.1


class PlanarPlant:
    """Two rotary joints with bounded commands and a finite step response."""

    def __init__(self, config: PlantConfig = PlantConfig()):
        self.config = config
        self.q = np.zeros(2, dtype=float)
        self.time_s = 0.0

    def reset(self, q_rad=(0.0, 0.0)):
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (2,) or not np.all(np.isfinite(q)) or np.any(q < self.config.joint_min_rad) or np.any(q > self.config.joint_max_rad):
            raise ValueError("reset requires two finite angles within joint limits")
        self.q = q.copy()
        self.time_s = 0.0
        return self.tip_m()

    def tip_m(self):
        q1, q2 = self.q
        l1, l2 = self.config.link_lengths_m
        return np.array([l1 * np.cos(q1) + l2 * np.cos(q1 + q2),
                         l1 * np.sin(q1) + l2 * np.sin(q1 + q2)])

    def step(self, increment_rad):
        command = np.asarray(increment_rad, dtype=float)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError("command requires two finite increments")
        applied = np.clip(command, -self.config.max_increment_rad, self.config.max_increment_rad)
        old_q = self.q.copy()
        self.q = np.clip(old_q + self.config.response_fraction * applied,
                         self.config.joint_min_rad, self.config.joint_max_rad)
        self.time_s += self.config.step_duration_s
        return {"tip_m": self.tip_m(), "command_rad": command.copy(),
                "bounded_command_rad": applied, "executed_rad": self.q - old_q,
                "saturated": bool(np.any(command != applied)), "time_s": self.time_s}
