"""Pure bounded-action state machine; it has no ROS dependencies."""

import math
from dataclasses import dataclass


DRIVE_TOLERANCE_M = 0.015
TURN_TOLERANCE_RAD = math.radians(1.0)


@dataclass(frozen=True)
class Limits:
    """Runtime safety and motion limits."""

    maximum_linear_speed_mps: float = 0.15
    maximum_angular_speed_radps: float = 0.60
    obstacle_stop_distance_m: float = 0.25
    command_timeout_s: float = 10.0
    odometry_staleness_timeout_s: float = 0.5
    scan_staleness_timeout_s: float = 0.5


@dataclass(frozen=True)
class Event:
    """One status transition emitted by the machine."""

    command_id: str
    state: str
    reason: str = ''


def wrapped_delta(current, previous):
    """Return the shortest signed angle from previous to current."""
    return math.atan2(
        math.sin(current - previous), math.cos(current - previous))


class ActionMachine:
    """Own one action and return an explicit velocity on every transition."""

    def __init__(self, limits):
        """Initialize an idle machine."""
        self.limits = limits
        self.pose = None
        self.odom_time = None
        self.scan = None
        self.scan_time = None
        self.active = None

    def update_odometry(self, x, y, yaw, now):
        """Store fresh pose feedback and accumulate signed turn progress."""
        self.pose = (x, y, yaw)
        self.odom_time = now
        if self.active and self.active['action'] == 'turn':
            self.active['turned'] += wrapped_delta(
                yaw, self.active['last_yaw'])
            self.active['last_yaw'] = yaw

    def update_scan(self, front, rear, all_around, now):
        """Store sector clearances and their observation time."""
        # None means no usable samples; +inf is valid clear space.
        self.scan = (front, rear, all_around)
        self.scan_time = now

    def submit(self, command, now):
        """Accept, reject, or preempt a command."""
        if command.action == 'stop':
            events = []
            if self.active:
                events.append(Event(self.active['id'], 'aborted', 'stopped'))
                self.active = None
            events.extend([
                Event(command.command_id, 'accepted'),
                Event(command.command_id, 'succeeded', 'stopped'),
            ])
            return events, (0.0, 0.0)
        if self.active:
            duplicate = command.command_id == self.active['id']
            reason = 'duplicate_active_id' if duplicate else 'busy'
            event = Event(command.command_id, 'rejected', reason)
            return [event], (0.0, 0.0)
        unavailable = self._unavailable(now)
        if unavailable:
            event = Event(command.command_id, 'rejected', unavailable)
            return [event], (0.0, 0.0)
        self.active = {
            'id': command.command_id,
            'action': command.action,
            'target': (command.value if command.action == 'drive'
                       else math.radians(command.value)),
            'start': now,
            'start_xy': self.pose[:2],
            'last_yaw': self.pose[2],
            'turned': 0.0,
        }
        accepted = Event(command.command_id, 'accepted')
        if self._blocked():
            self.active = None
            aborted = Event(command.command_id, 'aborted', 'obstacle')
            return [accepted, aborted], (0.0, 0.0)
        return [accepted], self._velocity()

    def tick(self, now):
        """Advance the action using current feedback and time."""
        if not self.active:
            return [], (0.0, 0.0)
        reason = self._unavailable(now)
        elapsed = now - self.active['start']
        if not reason and elapsed > self.limits.command_timeout_s:
            reason = 'timeout'
        if not reason and self._blocked():
            reason = 'obstacle'
        if reason:
            event = Event(self.active['id'], 'aborted', reason)
            self.active = None
            return [event], (0.0, 0.0)
        if self._remaining() <= self._tolerance():
            event = Event(self.active['id'], 'succeeded')
            self.active = None
            return [event], (0.0, 0.0)
        return [], self._velocity()

    def shutdown(self):
        """Abort any action and request a final zero velocity."""
        events = []
        if self.active:
            events.append(Event(self.active['id'], 'aborted', 'stopped'))
            self.active = None
        return events, (0.0, 0.0)

    def _unavailable(self, now):
        odometry_stale = (
            self.odom_time is None
            or now - self.odom_time
            > self.limits.odometry_staleness_timeout_s
        )
        if odometry_stale:
            return 'stale_odometry'
        scan_stale = (
            self.scan_time is None
            or now - self.scan_time > self.limits.scan_staleness_timeout_s
        )
        if scan_stale:
            return 'stale_scan'
        if any(value is None for value in self.scan):
            return 'stale_scan'
        return ''

    def _blocked(self):
        front, rear, all_around = self.scan
        action, target = self.active['action'], self.active['target']
        if action == 'turn':
            clearance = all_around
        else:
            clearance = front if target > 0 else rear
        return clearance <= self.limits.obstacle_stop_distance_m

    def _remaining(self):
        if self.active['action'] == 'drive':
            x0, y0 = self.active['start_xy']
            traveled = math.hypot(self.pose[0] - x0, self.pose[1] - y0)
        else:
            direction = 1.0 if self.active['target'] > 0.0 else -1.0
            traveled = max(0.0, direction * self.active['turned'])
        return max(0.0, abs(self.active['target']) - traveled)

    def _tolerance(self):
        if self.active['action'] == 'drive':
            return DRIVE_TOLERANCE_M
        return TURN_TOLERANCE_RAD

    def _velocity(self):
        remaining = self._remaining()
        sign = 1.0 if self.active['target'] > 0 else -1.0
        if self.active['action'] == 'drive':
            speed = min(
                self.limits.maximum_linear_speed_mps,
                max(0.03, remaining * 0.8),
            )
            return sign * speed, 0.0
        speed = min(
            self.limits.maximum_angular_speed_radps,
            max(0.12, remaining * 1.2),
        )
        return 0.0, sign * speed
