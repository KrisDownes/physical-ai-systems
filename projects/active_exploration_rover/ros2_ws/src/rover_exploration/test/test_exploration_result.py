"""Tests for the /exploration_result JSON schema and counter ownership."""

import json
from unittest.mock import MagicMock

from rover_exploration import mission_evaluator as me
from rover_exploration.frontier_node import FrontierDetector


def make_detector():
    detector = FrontierDetector.__new__(FrontierDetector)
    detector.recovery_cycle = MagicMock()
    detector.result_messages = []

    class FakePublisher:
        def __init__(self, sink):
            self.sink = sink

        def publish(self, message):
            self.sink.append(message)

    detector.exploration_complete_publisher = FakePublisher([])
    detector.exploration_result_publisher = FakePublisher(
        detector.result_messages
    )
    detector.exploration_complete = False
    detector.goals_assigned = 0
    detector.goals_reached = 0
    detector.failure_events = 0
    detector.temporary_failure_events = 0
    detector.recovery_requests = 0
    detector.permanent_failed_regions = []
    detector.visited_goal_regions = []
    detector.frontier_cells = set()
    detector.frontier_clusters = []
    detector.clock_s = 100.0
    detector.node_time_s = lambda: detector.clock_s
    detector.get_logger = lambda: MagicMock()
    # V16.4 terminal-outcome attributes (truthful completion vs blocked).
    detector.terminal_outcome = None
    detector.terminal_blocked_reason = None
    detector.terminal_geometric_frontier_cells = 0
    detector.terminal_geometric_frontier_clusters = 0
    detector.terminal_reachable_candidate_clusters = 0
    detector.terminal_post_exclusion_eligible = 0
    detector.terminal_temporary_rejected = 0
    detector.terminal_permanent_rejected = 0
    # Attributes the real stuck_check_callback / request_recovery paths
    # touch, so make_detector can exercise those methods directly.
    detector.recovery_request_publisher = MagicMock()
    detector.permanent_after_failures = 2
    detector.goal_path_failure_count = 0
    detector.reset_goal_progress = MagicMock()
    detector.log_failure = MagicMock()
    detector.latest_pose = (0.0, 0.0, 0.0)
    return detector


EXPECTED_KEYS = [
    'schema_version',
    'completed',
    'outcome',
    'blocked_reason',
    'completion_time_s',
    'goals_assigned',
    'goals_reached',
    'failure_events',
    'temporary_failure_events',
    'permanent_failed_regions',
    'recovery_requests',
    'visited_regions',
    'frontier_cells',
    'frontier_clusters',
    'geometric_frontier_cells',
    'geometric_frontier_clusters',
    'reachable_candidate_clusters',
    'post_exclusion_eligible',
]


def test_result_schema_exact_and_deterministic():
    detector = make_detector()
    detector.set_exploration_complete(True)
    payload = json.loads(detector.result_messages[0].data)
    # json.dumps(sort_keys=True) yields alphabetically sorted keys.
    assert sorted(payload.keys()) == sorted(EXPECTED_KEYS)
    assert set(payload.keys()) == set(EXPECTED_KEYS)
    assert detector.result_messages[0].data == json.dumps(
        payload, sort_keys=True
    )
    assert payload['schema_version'] == 2
    assert payload['completed'] is True
    assert set(payload) == set(me.RESULT_KEYS_V2)


def test_result_published_once_on_true_transition():
    # The structured result is emitted exactly once, via the real
    # set_exploration_complete transition, and never on a false transition.
    detector = make_detector()
    detector.set_exploration_complete(True)
    assert len(detector.result_messages) == 1
    detector.set_exploration_complete(False)
    detector.set_exploration_complete(True)
    assert len(detector.result_messages) == 2


def test_result_never_empty_on_false_transition():
    # Returning to an incomplete state must NOT publish an empty result.
    detector = make_detector()
    detector.set_exploration_complete(True)
    detector.result_messages.clear()
    detector.set_exploration_complete(False)
    assert detector.result_messages == []


def test_recovery_count_only_when_published():
    # Recovery counter increments only through the real request_recovery
    # path and only when the recovery coordination actually publishes.
    detector = make_detector()
    detector.recovery_cycle.publish_request.return_value = True
    before = detector.recovery_requests
    detector.request_recovery()
    assert detector.recovery_request_publisher.publish.call_count == 1
    assert detector.recovery_requests == before + 1

    detector.recovery_request_publisher.reset_mock()
    detector.recovery_cycle.publish_request.return_value = False
    detector.request_recovery()
    assert detector.recovery_request_publisher.publish.call_count == 0
    assert detector.recovery_requests == before + 1


def test_failure_registration_through_real_stuck_path():
    # The failure counters increment through the real stuck_check_callback
    # -> record_failure path, not by direct assignment.
    from collections import deque

    detector = make_detector()
    detector.recovery_cycle.planning_blocked = False
    detector.committed_goal_world = (5.0, 5.0)
    detector.latest_pose = (0.0, 0.0, 0.0)
    detector.node_time_s = lambda: 12.6
    detector.progress_samples = deque([
        (10.0, 0.0, 0.0, 0.0),
        (12.6, 0.0, 0.0, 0.0),
    ])
    detector.stuck_window_s = 4.0
    detector.stuck_progress_threshold_m = 0.05
    detector.stuck_alignment_threshold_rad = 0.2
    detector.blacklist_radius_m = 0.5
    detector.blacklist_duration_s = 10.0
    detector.map_resolution = 0.05
    detector.permanent_exclusion_radius_m = 0.20
    detector.failure_records = []
    detector.map_origin = (0.0, 0.0)
    before = detector.failure_events
    detector.stuck_check_callback()
    assert detector.failure_events == before + 1
    assert detector.temporary_failure_events == before + 1


def test_result_reflects_node_counters():
    # The result JSON reflects the node's real counter state at completion.
    detector = make_detector()
    detector.goals_assigned = 4
    detector.goals_reached = 4
    detector.recovery_requests = 2
    detector.failure_events = 1
    detector.temporary_failure_events = 1
    detector.permanent_failed_regions = [(0.0, 0.0)]
    detector.visited_goal_regions = [(1.0, 1.0), (2.0, 2.0)]
    detector.frontier_cells = {(0, 0), (1, 1)}
    detector.frontier_clusters = [3, 2]
    detector.set_exploration_complete(True)
    payload = json.loads(detector.result_messages[0].data)
    assert payload['goals_assigned'] == 4
    assert payload['goals_reached'] == 4
    assert payload['recovery_requests'] == 2
    assert payload['failure_events'] == 1
    assert payload['temporary_failure_events'] == 1
    assert payload['permanent_failed_regions'] == 1
    assert payload['visited_regions'] == 2
    assert payload['frontier_cells'] == 2
    assert payload['frontier_clusters'] == 2


def test_permanent_failed_regions_reflects_list_length():
    detector = make_detector()
    detector.permanent_failed_regions = [(0.0, 0.0), (1.0, 1.0)]
    detector.set_exploration_complete(True)
    payload = json.loads(detector.result_messages[0].data)
    assert payload['permanent_failed_regions'] == 2
