"""V15.3 correction: block premature completion while a frontier is in temporary cooldown."""

from collections import deque

from nav_msgs.msg import OccupancyGrid

from rover_exploration import frontier_memory as fm
from rover_exploration.frontier_node import FrontierDetector


FREE = 0
OCCUPIED = 100
UNKNOWN = -1

RESOLUTION = 0.05
ORIGIN_X = 0.0
ORIGIN_Y = 0.0


def _make_grid(width, height, data=None):
    if data is None:
        data = [FREE] * (width * height)
    msg = OccupancyGrid()
    msg.info.width = width
    msg.info.height = height
    msg.info.resolution = RESOLUTION
    msg.info.origin.position.x = ORIGIN_X
    msg.info.origin.position.y = ORIGIN_Y
    msg.data = list(data)
    return msg


def _cell_world(row, column):
    # Mirror grid_cell_center (cell-center convention).
    return (
        ORIGIN_X + (column + 0.5) * RESOLUTION,
        ORIGIN_Y + (row + 0.5) * RESOLUTION,
    )


class CompletionHarness:
    """Drive FrontierDetector.update_goal_and_path with a synthetic map."""

    def __init__(self, period_s=8.0):
        detector = FrontierDetector.__new__(FrontierDetector)

        # Completion / mission state.
        detector.completion_debounce_active = False
        detector.completion_debounce_started_s = 0.0
        detector.exploration_complete_logged = False
        detector.completion_debounce_period_s = period_s
        detector.completion_deferred_by_cooldown = False
        detector.exploration_complete = False
        detector.exploration_complete_publisher = type(
            'Pub', (), {'publish': lambda self, m: None}
        )()
        detector.exploration_result_publisher = type(
            'Pub', (), {'publish': lambda self, m: None}
        )()

        # Goal memory.
        detector.visited_goal_regions = []
        detector.permanent_failed_regions = []
        detector.failure_records = []

        # Mission counters.
        detector.goals_assigned = 0
        detector.goals_reached = 0
        detector.failure_events = 0
        detector.temporary_failure_events = 0
        detector.recovery_requests = 0

        # Candidate funnel diagnostics (reset each cycle by the method).
        detector.frontier_cells = set()
        detector.frontier_clusters = []
        detector.last_temporary_rejected_count = 0
        detector.last_permanent_rejected_count = 0
        detector.selected_frontier_cell = None
        detector.current_grid_path = None
        detector.last_planning_grid = None
        detector.escape_corridor_cells = []
        detector.committed_goal_world = None
        detector.goal_path_failure_count = 0
        detector.last_selected_cluster_size = 0
        detector.progress_samples = deque()

        # Planner parameters (safe defaults).
        detector.approach_search_radius_m = 2.0
        detector.blacklist_radius_m = 0.75
        detector.visited_radius_m = 0.60
        detector.goal_reached_distance_m = 0.30
        detector.permanent_after_failures = 2
        detector.blacklist_duration_s = 30.0
        detector.distance_slack_m = 0.10
        detector.maximum_goal_path_failures = 3
        detector.wall_closing_radius_m = 0.0
        detector.unknown_clearance_m = 0.0
        detector.rover_length_m = 0.30
        detector.rover_width_m = 0.25
        detector.path_clearance_m = 0.05

        # Recovery coordination (real state machine, nothing pending).
        detector.recovery_cycle = __import__(
            'rover_exploration.recovery_coordination',
            fromlist=['RecoveryCoordinationState'],
        ).RecoveryCoordinationState()

        self.clock_s = 0.0
        detector.node_time_s = lambda: self.clock_s

        class Logger:
            def __init__(self):
                self.messages = []

            def warning(self, message):
                self.messages.append(message)

            def info(self, message):
                self.messages.append(message)

            def debug(self, message):
                self.messages.append(message)

        self.logger = Logger()
        detector.get_logger = lambda: self.logger

        self.detector = detector

    def set_cluster(self, cells):
        # cells: list of (row, column). Must have >= 5 for a selectable
        # cluster (min_cluster_size default is 5 in the node).
        self.detector.frontier_clusters = [list(cells)]

    def call(self, map_message, robot_grid_cell=(1, 1)):
        self.detector.robot_grid_cell = robot_grid_cell
        raw = list(map_message.data)
        planning = list(map_message.data)
        self.detector.update_goal_and_path(
            map_message,
            raw_data=raw,
            planning_data=planning,
            robot_x=_cell_world(*robot_grid_cell)[0],
            robot_y=_cell_world(*robot_grid_cell)[1],
        )


# A 5-cell frontier cluster on a small free map, well away from the robot.
CLUSTER = [(10, 10), (10, 11), (10, 12), (10, 13), (10, 14)]
CLUSTER_WORLD = _cell_world(10, 12)  # centroid column -> world


def test_temporary_cooldown_rejects_but_defers_completion():
    # A reachable frontier candidate inside an active temporary
    # cooldown is rejected as temporary, does NOT start/advance the
    # completion debounce, does NOT publish completion, and issues no
    # movement command.
    harness = CompletionHarness()
    detector = harness.detector
    harness.set_cluster(CLUSTER)

    detector.failure_records = [
        {
            'x': CLUSTER_WORLD[0],
            'y': CLUSTER_WORLD[1],
            'failure_count': 1,
            'blocked_until_s': harness.clock_s + 30.0,
        }
    ]

    width = height = 20
    grid = _make_grid(width, height)

    harness.call(grid)

    assert detector.last_temporary_rejected_count == 1
    assert detector.last_permanent_rejected_count == 0
    assert detector.exploration_complete is False
    # Debounce must NOT have started.
    assert detector.completion_debounce_active is False
    assert detector.completion_debounce_started_s == 0.0
    # No goal assigned, no movement command object.
    assert detector.goals_assigned == 0
    assert not hasattr(detector, 'command_publisher')
    # Deferral should be recorded.
    assert detector.completion_deferred_by_cooldown is True


def test_temporary_cooldown_holds_completion_past_debounce():
    # Repeat map callbacks for longer than the 8s debounce while the
    # 30s cooldown remains active: /exploration_complete stays false.
    harness = CompletionHarness(period_s=8.0)
    detector = harness.detector
    harness.set_cluster(CLUSTER)

    detector.failure_records = [
        {
            'x': CLUSTER_WORLD[0],
            'y': CLUSTER_WORLD[1],
            'failure_count': 1,
            'blocked_until_s': 30.0,
        }
    ]

    grid = _make_grid(20, 20)

    for t in (0.0, 1.0, 5.0, 8.1, 12.0, 20.0, 29.9):
        harness.clock_s = t
        harness.call(grid)
        assert detector.exploration_complete is False, (
            f'completion flipped true at t={t}'
        )
        assert detector.completion_debounce_active is False


def test_cooldown_expiry_retries_and_assigns_goal():
    # Advance time past cooldown expiry: the failure record is pruned,
    # the same candidate becomes eligible, a goal is assigned, and
    # goals_assigned increments exactly once.
    harness = CompletionHarness()
    detector = harness.detector
    harness.set_cluster(CLUSTER)

    detector.failure_records = [
        {
            'x': CLUSTER_WORLD[0],
            'y': CLUSTER_WORLD[1],
            'failure_count': 1,
            'blocked_until_s': 30.0,
        }
    ]

    grid = _make_grid(20, 20)

    # While cooldown active: deferred, no assignment.
    harness.clock_s = 10.0
    harness.call(grid)
    assert detector.goals_assigned == 0
    assert detector.completion_deferred_by_cooldown is True

    # Past expiry: prune_expired_cooldowns clears the cooldown inside
    # update_goal_and_path, candidate becomes eligible, goal assigned.
    harness.clock_s = 31.0
    harness.call(grid)
    assert detector.goals_assigned == 1
    assert detector.last_temporary_rejected_count == 0
    assert detector.last_permanent_rejected_count == 0
    assert detector.completion_deferred_by_cooldown is False
    assert detector.committed_goal_world is not None
    assert detector.exploration_complete is False


def test_unreachable_only_still_completes_after_debounce():
    # A map containing only unreachable frontiers: the existing
    # completion debounce still starts and completion becomes true
    # after 8 seconds.
    harness = CompletionHarness(period_s=8.0)
    detector = harness.detector
    harness.set_cluster(CLUSTER)

    width = height = 20
    data = [FREE] * (width * height)
    # Solid occupied wall at column 5 sealing the cluster's room away
    # from the rover at column 1.
    for row in range(height):
        data[row * width + 5] = OCCUPIED
    grid = _make_grid(width, height, data)

    # First callback starts the debounce.
    harness.call(grid)
    assert detector.completion_debounce_active is True
    assert detector.last_temporary_rejected_count == 0
    assert detector.exploration_complete is False

    # Before the debounce elapses: still not complete.
    harness.clock_s = 7.9
    harness.call(grid)
    assert detector.exploration_complete is False

    # After the debounce elapses: completion declared.
    harness.clock_s = 8.1
    harness.call(grid)
    assert detector.exploration_complete is True


def test_visited_only_preserves_existing_completion_behavior():
    # A map whose only candidate is already visited: terminal
    # completion behavior is preserved (debounce runs, completes).
    harness = CompletionHarness(period_s=8.0)
    detector = harness.detector
    harness.set_cluster(CLUSTER)
    detector.visited_goal_regions = [CLUSTER_WORLD]

    grid = _make_grid(20, 20)

    harness.call(grid)
    assert detector.last_temporary_rejected_count == 0
    assert detector.last_permanent_rejected_count == 0
    assert detector.completion_debounce_active is True

    harness.clock_s = 8.1
    harness.call(grid)
    assert detector.exploration_complete is True


def test_promotion_to_permanent_behavior_preserved():
    # A second failure after cooldown must still promote to permanent;
    # V15.3 does not weaken permanent-blacklist semantics. record_failure
    # is unchanged, but we assert the contract explicitly.
    failure_records = []
    permanent_regions = []

    # First failure -> temporary cooldown.
    out1 = fm.record_failure(
        failure_records=failure_records,
        permanent_regions=permanent_regions,
        x=CLUSTER_WORLD[0],
        y=CLUSTER_WORLD[1],
        now_s=0.0,
        match_radius_m=0.75,
        blacklist_duration_s=30.0,
        promotion_failures=2,
    )
    assert out1 == 'new'
    assert failure_records[0]['blocked_until_s'] == 30.0

    # Second failure (even after cooldown expiry) -> promoted.
    out2 = fm.record_failure(
        failure_records=failure_records,
        permanent_regions=permanent_regions,
        x=CLUSTER_WORLD[0],
        y=CLUSTER_WORLD[1],
        now_s=100.0,
        match_radius_m=0.75,
        blacklist_duration_s=30.0,
        promotion_failures=2,
    )
    assert out2 == 'promoted'
    assert failure_records[0]['blocked_until_s'] == float('inf')
    assert len(permanent_regions) == 1
    # is_excluded must now report permanent (never resurrected).
    assert (
        fm.is_excluded(
            x=CLUSTER_WORLD[0],
            y=CLUSTER_WORLD[1],
            failure_records=failure_records,
            permanent_regions=permanent_regions,
            visited_regions=[],
            now_s=100.0,
            exclusion_radius_m=0.75,
        )
        == 'permanent'
    )
