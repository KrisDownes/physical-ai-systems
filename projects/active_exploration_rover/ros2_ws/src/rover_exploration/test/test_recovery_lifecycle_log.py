"""Regression tests for the recovery lifecycle logging."""

from unittest.mock import MagicMock

from rover_exploration.frontier_node import FrontierDetector


def make_detector():
    """Build a detector with ROS bits stubbed out."""
    detector = FrontierDetector.__new__(FrontierDetector)

    from collections import deque

    detector.recovery_cycle = __import__(
        'rover_exploration.recovery_coordination',
        fromlist=['RecoveryCoordinationState'],
    ).RecoveryCoordinationState()

    detector.progress_samples = deque()
    detector.reset_goal_progress_calls = []

    def reset_goal_progress():
        detector.progress_samples.clear()
        detector.reset_goal_progress_calls.append(True)

    detector.reset_goal_progress = reset_goal_progress
    detector.get_logger = lambda: MagicMock()

    return detector


class Status:
    def __init__(self, data):
        self.data = data


def test_active_status_does_not_log_cycle_end():
    detector = make_detector()

    detector.recovery_status_callback(Status(True))

    assert detector.recovery_cycle.recovery_active is True
    assert detector.recovery_cycle.planning_blocked is True
    assert detector.reset_goal_progress_calls == []


def test_inactive_after_active_resets_window_once():
    detector = make_detector()

    detector.recovery_status_callback(Status(True))
    detector.recovery_status_callback(Status(False))

    # Exactly one window reset for the completed cycle.
    assert len(detector.reset_goal_progress_calls) == 1
    assert detector.recovery_cycle.planning_blocked is False

    # A second inactive status without a new cycle does nothing.
    detector.recovery_status_callback(Status(False))
    assert len(detector.reset_goal_progress_calls) == 1


def test_inactive_after_pending_request_also_resets():
    detector = make_detector()
    detector.request_recovery = MagicMock()

    published = (
        detector.recovery_cycle.publish_request()
    )

    assert published is True
    assert detector.recovery_cycle.planning_blocked is True

    detector.recovery_status_callback(Status(False))

    assert len(detector.reset_goal_progress_calls) == 1
    assert detector.recovery_cycle.planning_blocked is False
