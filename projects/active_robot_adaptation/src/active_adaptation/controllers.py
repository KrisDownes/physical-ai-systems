"""Controllers receive only observed coordinates and their own sent commands.

J maps commanded joint increments (rad) to camera displacement (camera units).
It starts from a shared pre-change calibration, not online simulator truth.
"""
import numpy as np


class JacobianController:
    def __init__(self, initial_jacobian, adapt=False, forgetting=0.98, damping=0.03):
        self.j = np.array(initial_jacobian, dtype=float, copy=True)
        self.p = np.eye(2) * 10.0
        self.adapt = adapt
        self.forgetting = forgetting
        self.damping = damping
        self.previous_observation = None
        self.previous_command = None
        self.command_saturated = False

    def act(self, observation, target, time_s):
        observation = np.asarray(observation)
        if self.adapt and self.previous_command is not None:
            u = self.previous_command
            delta = observation - self.previous_observation
            pu = self.p @ u
            gain = pu / (self.forgetting + u @ pu)
            self.j += np.outer(delta - self.j @ u, gain)
            self.p = (self.p - np.outer(gain, u @ self.p)) / self.forgetting
        error = np.asarray(target) - observation
        command = self.j.T @ np.linalg.solve(self.j @ self.j.T + self.damping**2 * np.eye(2), 0.7 * error)
        # Controller explicitly limits its sent command; both methods share this rule.
        bounded = np.clip(command, -0.1, 0.1)
        self.command_saturated = bool(np.any(bounded != command))
        command = bounded
        self.previous_observation = observation.copy()
        self.previous_command = command.copy()
        return command


def model_jacobian(q, lengths=(0.3, 0.2), response=0.8):
    """Nominal pre-change camera and actuator model, camera units per radian."""
    a, b = q
    l1, l2 = lengths
    return response * np.array([[-l1*np.sin(a)-l2*np.sin(a+b), -l2*np.sin(a+b)],
                                [l1*np.cos(a)+l2*np.cos(a+b), l2*np.cos(a+b)]])


class ModelController(JacobianController):
    """Dead-reckon configuration from commands and a nominal pre-change model.

    Assumes known nominal lengths, camera map and response; chooses the negative
    elbow branch from the first observation. No joint-state channel is read.
    """
    def __init__(self, initial_jacobian):
        super().__init__(initial_jacobian)
        self.q_estimate = None

    def act(self, observation, target, time_s):
        if self.q_estimate is None:
            x, y = observation
            l1, l2 = 0.3, 0.2
            cosine = (x*x + y*y - l1*l1 - l2*l2) / (2*l1*l2)
            if not -1.0 - 1e-10 <= cosine <= 1.0 + 1e-10:
                raise ValueError('initial observation outside nominal workspace')
            q2 = -np.arccos(np.clip(cosine, -1, 1))
            q1 = np.arctan2(y, x) - np.arctan2(l2*np.sin(q2), l1+l2*np.cos(q2))
            self.q_estimate = np.array([q1, q2])
        elif self.previous_command is not None:
            self.q_estimate = np.clip(self.q_estimate + 0.8 * self.previous_command, -np.pi, np.pi)
        self.j = model_jacobian(self.q_estimate)
        return super().act(observation, target, time_s)
