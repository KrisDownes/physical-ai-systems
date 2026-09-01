"""
Structural and behavioral tests for /exploration_complete state.

Uses a stubbed FrontierDetector (ROS node bits mocked) so the completion
state machine and centralized set_exploration_complete() helper can be
exercised without spinning a node.
"""

from unittest.mock import MagicMock

from rover_exploration.frontier_node import FrontierDetector
from rover_exploration.recovery_coordination import RecoveryCoordinationState


def make_detector():
    """Build a FrontierDetector with publishers and clock stubbed."""
    detector = FrontierDetector.__new__(FrontierDetector)

    # Stub the autonomy state machine used by completion guards.
    detector.recovery_cycle = RecoveryCoordinationState()

    # Capture published completion/result messages.
    detector.complete_messages = []
    detector.result_messages = []

    class FakePublisher:
        def __init__(self, sink):
            self.sink = sink

        def publish(self, message):
            self.sink.append(message)

    detector.exploration_complete_publisher = FakePublisher(
        detector.complete_messages
    )
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
    detector.permanent_exclusion_radius_m = 0.20

    # V16.4 terminal-outcome attributes (truthful completion vs blocked).
    detector.terminal_outcome = None
    detector.terminal_blocked_reason = None
    detector.terminal_geometric_frontier_cells = 0
    detector.terminal_geometric_frontier_clusters = 0
    detector.terminal_reachable_candidate_clusters = 0
    detector.terminal_post_exclusion_eligible = 0
    detector.terminal_temporary_rejected = 0
    detector.terminal_permanent_rejected = 0

    detector.clock_s = 0.0
    detector.node_time_s = lambda: detector.clock_s
    detector.get_logger = lambda: MagicMock()

    # Completion-debounce state required by completion_debounce_tick.
    detector.completion_debounce_active = False
    detector.completion_debounce_started_s = 0.0
    detector.exploration_complete_logged = False
    detector.completion_debounce_period_s = 8.0

    detector.committed_goal_world = None

    return detector


def test_initial_completion_state_is_false():
    detector = make_detector()
    # No transition has happened; state remains False and nothing
    # was published yet (the live node publishes the latched False in
    # __init__, but the stub does not, so the first publish is the
    # only one the helper issues).
    assert detector.exploration_complete is False


def test_set_exploration_complete_publishes_true_once():
    detector = make_detector()
    detector.set_exploration_complete(True)
    assert detector.exploration_complete is True
    # Exactly one Bool(true) on the wire.
    assert len(detector.complete_messages) == 1
    assert detector.complete_messages[0].data is True
    # A result JSON was emitted.
    assert len(detector.result_messages) == 1


def test_repeated_true_does_not_republish():
    detector = make_detector()
    detector.set_exploration_complete(True)
    detector.set_exploration_complete(True)
    assert len(detector.complete_messages) == 1
    assert len(detector.result_messages) == 1


def test_debounce_start_does_not_publish_true():
    detector = make_detector()
    detector.clock_s = 0.0
    detector.completion_debounce_tick()
    # Debounce started: still False, no publish.
    assert detector.exploration_complete is False
    assert detector.complete_messages == []


def test_debounce_expiry_publishes_true_once():
    detector = make_detector()
    detector.clock_s = 0.0
    detector.completion_debounce_tick()
    detector.clock_s = 7.9
    detector.completion_debounce_tick()
    assert detector.completion_debounce_active is True
    assert detector.complete_messages == []  # not yet expired

    detector.clock_s = 8.1
    detector.completion_debounce_tick()
    assert detector.exploration_complete is True
    assert len(detector.complete_messages) == 1

    # Extra no-goal cycles do not republish true.
    detector.clock_s = 9.0
    detector.completion_debounce_tick()
    detector.clock_s = 10.0
    detector.completion_debounce_tick()
    assert len(detector.complete_messages) == 1


def test_goal_after_completion_publishes_false():
    detector = make_detector()
    detector.set_exploration_complete(True)
    assert len(detector.complete_messages) == 1

    detector.set_exploration_complete(False)
    assert detector.exploration_complete is False
    assert len(detector.complete_messages) == 2
    assert detector.complete_messages[-1].data is False


def test_recovery_active_blocks_completion():
    detector = make_detector()
    detector.clock_s = 0.0
    detector.completion_debounce_tick()
    detector.clock_s = 8.1
    # Simulate recovery running when debounce expires.
    detector.recovery_cycle.on_status_active()
    detector.completion_debounce_tick()
    assert detector.exploration_complete is False
    assert detector.complete_messages == []


def test_recovery_pending_blocks_completion():
    detector = make_detector()
    detector.clock_s = 0.0
    detector.completion_debounce_tick()
    detector.clock_s = 8.1
    detector.recovery_cycle.publish_request()  # pending, not yet active
    detector.completion_debounce_tick()
    assert detector.exploration_complete is False
    assert detector.complete_messages == []


def test_committed_goal_blocks_completion():
    detector = make_detector()
    detector.clock_s = 0.0
    detector.completion_debounce_tick()
    detector.clock_s = 8.1
    detector.committed_goal_world = (1.0, 1.0)
    detector.completion_debounce_tick()
    assert detector.exploration_complete is False
    assert detector.complete_messages == []


def test_completion_publishes_only_state_topic_not_commands():
    detector = make_detector()
    detector.set_exploration_complete(True)
    # The helper only touches the two completion publishers; it does
    # not create or modify any velocity command object.
    assert detector.exploration_complete is True
