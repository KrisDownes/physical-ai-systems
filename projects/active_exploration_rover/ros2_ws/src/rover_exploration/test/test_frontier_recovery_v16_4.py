"""
V16.4: bounded frontier recovery and truthful completion.

These tests pin the corrections that stop a failed approach from being
promoted into a whole-cluster permanent blacklist, stop "no selectable
frontier" from being reported as successful completion, and make the
terminal outcome truthful (success vs blocked).
"""

from collections import deque
from unittest.mock import MagicMock

from rover_exploration.frontier_memory import (
    is_excluded,
    record_failure,
)
import rover_exploration.frontier_node as frontier_node_module
from rover_exploration.frontier_node import FrontierDetector
from rover_exploration.grid_planning import (
    compute_reachable_component,
)


def make_detector():
    """
    Construct a FrontierDetector without rclpy init.

    Mirrors the existing test_exploration_result harness.
    """
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
    detector.failure_records = []
    detector.visited_goal_regions = []
    detector.frontier_cells = set()
    detector.frontier_clusters = []
    detector.committed_goal_world = None
    detector.goal_path_failure_count = 0
    detector.active_frontier_target_anchor = None
    detector.attempted_approach_cells = set()
    detector.attempted_approach_paths = set()
    detector.fresh_approach_count = 0
    detector.maximum_fresh_approaches_per_target = 3
    detector.last_selected_cluster_size = 0
    detector.terminal_outcome = None
    detector.terminal_blocked_reason = None
    detector.terminal_geometric_frontier_cells = 0
    detector.terminal_geometric_frontier_clusters = 0
    detector.terminal_reachable_candidate_clusters = 0
    detector.terminal_post_exclusion_eligible = 0
    detector.terminal_temporary_rejected = 0
    detector.terminal_permanent_rejected = 0
    detector.selected_frontier_cell = None
    detector.current_grid_path = None
    detector.approach_search_radius_m = 1.5
    detector.blacklist_radius_m = 0.75
    detector.blacklist_duration_s = 30.0
    detector.permanent_after_failures = 1
    detector.permanent_exclusion_radius_m = 0.20
    detector.map_resolution = 0.05
    detector.map_origin = (0.0, 0.0)
    detector.robot_grid_cell = (1, 1)
    detector.reset_goal_progress = MagicMock()
    detector.get_logger = lambda: MagicMock()
    detector.log_failure = MagicMock()
    # node_time_s is unused by decide_terminal_outcome / fresh_approach.
    detector.node_time_s = lambda: 0.0
    return detector


# ---- T4 / T5: truthful terminal outcome (success vs blocked) ----

def test_no_frontier_is_genuine_success():
    detector = make_detector()
    detector.decide_terminal_outcome(
        geometric_frontier_cells=0,
        geometric_frontier_clusters=0,
        reachable_candidate_clusters=0,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=0,
    )
    assert detector.terminal_outcome == 'success'
    assert detector.terminal_blocked_reason is None


def test_residual_frontier_all_blacklisted_is_blocked():
    detector = make_detector()
    detector.decide_terminal_outcome(
        geometric_frontier_cells=68,
        geometric_frontier_clusters=3,
        reachable_candidate_clusters=3,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=3,
    )
    assert detector.terminal_outcome == 'blocked'
    assert 'permanently blacklisted' in (
        detector.terminal_blocked_reason or ''
    )


def test_residual_frontier_visited_only_is_blocked():
    """
    A >= 5-cell frontier whose only rejects are visited/too-close cells.

    This is NOT success: those are stable rejections of an unresolved
    frontier, not transient cooldowns. Under the genuine contract,
    completion requires no residual >= 5-cell component.
    """
    detector = make_detector()
    detector.decide_terminal_outcome(
        geometric_frontier_cells=40,
        geometric_frontier_clusters=2,
        reachable_candidate_clusters=2,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=0,
    )
    assert detector.terminal_outcome == 'blocked'
    assert 'visited' in (detector.terminal_blocked_reason or '')


def test_residual_frontier_goal_distance_only_is_blocked():
    """
    Same as visited-only: a single >= 5-cell cluster rejected only.

    Because its approach is within goal-reached distance must not be
    reported as successful exploration.
    """
    detector = make_detector()
    detector.decide_terminal_outcome(
        geometric_frontier_cells=20,
        geometric_frontier_clusters=1,
        reachable_candidate_clusters=1,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=0,
    )
    assert detector.terminal_outcome == 'blocked'


def test_small_stray_frontier_without_large_component_is_success():
    """
    Only sub-threshold stray frontier cells (no >= 5-cell component).

    Under the current contract that is genuine completion, not a
    blocked terminal state.
    """
    detector = make_detector()
    detector.decide_terminal_outcome(
        geometric_frontier_cells=3,
        geometric_frontier_clusters=0,
        reachable_candidate_clusters=0,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=0,
    )
    assert detector.terminal_outcome == 'success'


def test_blocked_outcome_in_result_json():
    detector = make_detector()
    detector.decide_terminal_outcome(
        geometric_frontier_cells=152,
        geometric_frontier_clusters=3,
        reachable_candidate_clusters=3,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=3,
    )
    detector.set_exploration_complete(True)
    payload = __import__(
        'json'
    ).loads(detector.result_messages[0].data)
    assert payload['outcome'] == 'blocked'
    assert payload['completed'] is True
    assert payload['geometric_frontier_cells'] == 152
    assert payload['post_exclusion_eligible'] == 0


# ---- T8: safe reconsideration on relevant map change ----

def test_cooldown_expiry_re_enables_candidate():
    """
    Expiry re-enables a cooled-down candidate without losing the count.

    A temporary failure cools down; the candidate becomes eligible again
    (map change / time advanced) without resetting the lifetime count.
    This is the bounded-retry safety valve.
    """
    memory = {'records': [], 'permanent': [], 'visited': []}
    record_failure(
        failure_records=memory['records'],
        permanent_regions=memory['permanent'],
        x=1.0, y=1.0, now_s=0.0,
        match_radius_m=0.75,
        blacklist_duration_s=30.0,
        promotion_failures=2,
    )
    # Active cooldown now.
    assert is_excluded(
        x=1.0, y=1.0, failure_records=memory['records'],
        permanent_regions=memory['permanent'], visited_regions=[],
        now_s=10.0, exclusion_radius_m=0.75,
        visited_radius_m=0.60,
    ) == 'temporary'
    # After expiry the candidate is eligible again (count survives).
    assert is_excluded(
        x=1.0, y=1.0, failure_records=memory['records'],
        permanent_regions=memory['permanent'], visited_regions=[],
        now_s=40.0, exclusion_radius_m=0.75,
        visited_radius_m=0.60,
    ) is None


# ---- T3: a stale invalidated path does not masquerade as many attempts,
# and a fresh plan to the same cluster is preferred over promotion ----

def _build_grid(width, height):
    # All-free grid; rover at (1, 1).
    return [0] * (width * height)


def test_fresh_approach_found_for_failed_goal_same_cluster():
    """
    Fresh plan to the same cluster is preferred over blacklisting.

    When a committed goal's path is invalid, the node must try a fresh
    collision-checked plan to the SAME cluster before blacklisting it.
    With a reachable free cluster, a valid alternative approach is returned
    and no permanent region is promoted.
    """
    detector = make_detector()
    width, height = 31, 11
    raw = _build_grid(width, height)
    planning = _build_grid(width, height)
    bfs = compute_reachable_component(
        data=planning, width=width, height=height, start=(1, 1)
    )
    # A frontier cluster far from the rover but reachable.
    cluster = {(1, 20), (1, 21), (2, 20)}
    detector.frontier_clusters = [cluster]
    detector.approach_search_radius_m = 1.5
    detector.active_frontier_target_anchor = (1, 20)
    detector.attempted_approach_cells = {(1, 20)}
    detector.attempted_approach_paths = {((1, 1), (1, 20))}

    fresh = detector.fresh_approach_for_failed_goal(
        map_message=_fake_map(width, height),
        raw_data=raw, planning_data=planning, bfs=bfs,
        width=width, height=height,
    )
    assert fresh is not None
    cell, path = fresh
    assert path is not None
    # No permanent region was promoted by the fresh-plan attempt itself.
    assert detector.permanent_failed_regions == []


def test_fresh_approach_none_when_cluster_unreachable():
    """
    Unreachable cluster yields None so the caller promotes (scoped).

    When the cluster is genuinely unreachable, the fresh plan returns
    None so the caller promotes the failed approach (scoped, not blanket).
    """
    detector = make_detector()
    width, height = 21, 3
    raw = _build_grid(width, height)
    planning = _build_grid(width, height)
    # Wall splits the rover (left) from the cluster (right).
    for row in range(height):
        raw[row * width + 10] = 100
        planning[row * width + 10] = 100
    bfs = compute_reachable_component(
        data=planning, width=width, height=height, start=(1, 1)
    )
    cluster = {(1, 15), (1, 16)}
    detector.frontier_clusters = [cluster]
    detector.approach_search_radius_m = 1.5
    detector.active_frontier_target_anchor = (1, 15)

    fresh = detector.fresh_approach_for_failed_goal(
        map_message=_fake_map(width, height),
        raw_data=raw, planning_data=planning, bfs=bfs,
        width=width, height=height,
    )
    assert fresh is None


# ---- V16.4 retry-target invariants ----

def _fresh(detector, cluster, width=31, height=11):
    # Raw-free seeds are limited to the target cluster. This prevents the
    # bounded standoff search from discovering arbitrary surrounding cells.
    raw = [-1] * (width * height)
    for row, column in cluster:
        raw[row * width + column] = 0
    planning = _build_grid(width, height)
    bfs = compute_reachable_component(
        data=planning, width=width, height=height, start=(1, 1)
    )
    detector.frontier_clusters = [cluster]
    return detector.fresh_approach_for_failed_goal(
        map_message=_fake_map(width, height), raw_data=raw,
        planning_data=planning, bfs=bfs, width=width, height=height,
    )


def test_failed_approach_cell_is_never_its_own_replacement():
    detector = make_detector()
    cluster = {(1, 20)}
    detector.active_frontier_target_anchor = (1, 20)
    detector.attempted_approach_cells = {(1, 20)}
    assert _fresh(detector, cluster) is None


def test_identical_reconstructed_path_is_rejected():
    detector = make_detector()
    cluster = {(1, 20), (1, 21)}
    detector.active_frontier_target_anchor = (1, 20)
    repeated_path = ((1, 1), (1, 20))
    detector.attempted_approach_paths = {repeated_path}
    original = frontier_node_module.reconstruct_grid_path
    frontier_node_module.reconstruct_grid_path = lambda *_: list(repeated_path)
    try:
        assert _fresh(detector, cluster) is None
    finally:
        frontier_node_module.reconstruct_grid_path = original


def test_different_valid_approach_is_accepted_and_recorded():
    detector = make_detector()
    cluster = {(1, 20), (1, 21)}
    detector.active_frontier_target_anchor = (1, 20)
    detector.attempted_approach_cells = {(1, 20)}
    fresh = _fresh(detector, cluster)
    assert fresh is not None
    cell, path = fresh
    assert cell == (1, 21)
    detector.attempted_approach_cells.add(cell)
    detector.attempted_approach_paths.add(tuple(path))
    assert _fresh(detector, cluster) is None


def test_nearby_cluster_cannot_replace_selected_target():
    detector = make_detector()
    detector.active_frontier_target_anchor = (1, 20)
    detector.attempted_approach_cells = {(1, 20)}
    # B is within the approach radius but does not contain A's anchor.
    detector.frontier_clusters = [{(1, 20)}, {(1, 22), (1, 23)}]
    raw = [-1] * (31 * 11)
    raw[1 * 31 + 20] = 0
    raw[1 * 31 + 22] = 0
    raw[1 * 31 + 23] = 0
    planning = _build_grid(31, 11)
    bfs = compute_reachable_component(
        data=planning, width=31, height=11, start=(1, 1)
    )
    assert detector.fresh_approach_for_failed_goal(
        map_message=_fake_map(31, 11), raw_data=raw,
        planning_data=planning, bfs=bfs, width=31, height=11,
    ) is None


def test_alternative_cycle_is_finite_and_target_state_clears_on_change():
    detector = make_detector()
    cluster = {(1, 20), (1, 21), (1, 22)}
    detector.start_frontier_target_attempt(
        anchor=(1, 20), approach_cell=(1, 20), path=[(1, 20)],
    )
    while True:
        fresh = _fresh(detector, cluster)
        if fresh is None:
            break
        cell, path = fresh
        detector.attempted_approach_cells.add(cell)
        detector.attempted_approach_paths.add(tuple(path))
    assert detector.attempted_approach_cells == cluster
    detector.start_frontier_target_attempt(
        anchor=(5, 5), approach_cell=(5, 4), path=[(5, 4)],
    )
    assert detector.active_frontier_target_anchor == (5, 5)
    assert detector.attempted_approach_cells == {(5, 4)}
    assert detector.goal_path_failure_count == 0


def test_explicit_fresh_approach_limit_rejects_remaining_alternatives():
    detector = make_detector()
    detector.maximum_fresh_approaches_per_target = 2
    detector.fresh_approach_count = 2
    detector.active_frontier_target_anchor = (1, 20)
    # This target has six valid alternatives, but the explicit cap wins.
    cluster = {(1, column) for column in range(20, 27)}
    assert _fresh(detector, cluster) is None


def test_exhaustion_registers_scoped_failure_and_reaches_blocked_flow():
    detector = make_detector()
    detector.committed_goal_world = (1.025, 1.025)
    detector.start_frontier_target_attempt(
        anchor=(1, 20), approach_cell=(1, 20), path=[(1, 20)],
    )
    # No unattempted approach remains, so the caller's fall-through uses the
    # same scoped failure-registration method used by update_goal_and_path.
    assert _fresh(detector, {(1, 20)}) is None
    detector.abandon_path_invalid_target(1.025, 1.025, now_s=0.0)
    # Promotion remains governed by the existing two-hit memory policy.
    detector.abandon_path_invalid_target(1.025, 1.025, now_s=31.0)
    assert detector.permanent_failed_regions == [(1.025, 1.025, 0.20)]
    assert detector.failure_events == 2
    assert detector.active_frontier_target_anchor is None
    detector.decide_terminal_outcome(
        geometric_frontier_cells=5,
        geometric_frontier_clusters=1,
        reachable_candidate_clusters=1,
        post_exclusion_eligible=0,
        temporary_rejected=0,
        permanent_rejected=1,
    )
    assert detector.terminal_outcome == 'blocked'


def test_temporary_stuck_recovery_keeps_target_retry_budget():
    detector = make_detector()
    detector.progress_samples = deque()
    detector.latest_pose = (0.0, 0.0, 0.0)
    detector.recovery_cycle.planning_blocked = False
    detector.committed_goal_world = (1.0, 1.0)
    detector.stuck_window_s = 6.0
    detector.stuck_progress_threshold_m = 0.05
    detector.stuck_alignment_threshold_rad = 0.4
    detector.start_frontier_target_attempt(
        anchor=(10, 10), approach_cell=(10, 10), path=[(10, 10)],
    )
    detector.fresh_approach_count = 2
    detector.request_recovery = MagicMock()
    original = frontier_node_module.is_stuck
    frontier_node_module.is_stuck = lambda **_: True
    try:
        detector.stuck_check_callback()
    finally:
        frontier_node_module.is_stuck = original
    assert detector.fresh_approach_count == 2
    assert detector.active_frontier_target_anchor == (10, 10)
    assert detector.failure_records[0]['blocked_until_s'] == 30.0


# ---- helper ----

class _FakeMap:
    """Minimal stand-in for a nav_msgs/OccupancyGrid message."""

    def __init__(self, width, height):
        self.info = _Info(width, height)


class _Info:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.resolution = 0.05
        self.origin = _Origin()


class _Origin:
    position = type('P', (), {'x': 0.0, 'y': 0.0})()


def _fake_map(width, height):
    return _FakeMap(width, height)
