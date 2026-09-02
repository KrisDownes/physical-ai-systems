"""Enduring exploration-policy contract tests."""

from rover_exploration.exploration_policy import (
    ExplorationPolicy,
    PolicyConfig,
    TargetState,
)
from rover_exploration.grid_planning import compute_reachable_component


WIDTH = 21
HEIGHT = 11
ROBOT_CELL = (5, 1)
ROBOT_WORLD = (1.5, 5.5)


def make_policy(**overrides):
    values = {
        'goal_reached_distance_m': 0.25,
        'maximum_goal_path_failures': 1,
        'maximum_fresh_approaches_per_target': 3,
        'stuck_window_s': 6.0,
        'stuck_progress_threshold_m': 0.05,
        'stuck_alignment_threshold_rad': 0.39,
        'blacklist_radius_m': 0.75,
        'blacklist_duration_s': 30.0,
        'permanent_after_failures': 2,
        'visited_radius_m': 0.60,
        'permanent_exclusion_radius_m': 0.20,
        'distance_slack_m': 2.0,
        'completion_debounce_period_s': 8.0,
        'approach_search_radius_m': 5.0,
    }
    values.update(overrides)
    return ExplorationPolicy(PolicyConfig(**values))


def cycle(policy, clusters=(), now_s=0.0, planning_data=None):
    raw = [0] * (WIDTH * HEIGHT)
    planning = list(raw if planning_data is None else planning_data)
    cells = set().union(*clusters) if clusters else set()
    return policy.update(
        raw_data=raw,
        planning_data=planning,
        width=WIDTH,
        height=HEIGHT,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
        frontier_cells=cells,
        frontier_clusters=list(clusters),
        robot_cell=ROBOT_CELL,
        robot_world=ROBOT_WORLD,
        now_s=now_s,
    )


def scaled_cycle(policy, clusters, now_s, planning_data=None):
    """Run the real policy flow on a 0.1 m grid."""
    raw = [0] * (WIDTH * HEIGHT)
    planning = list(raw if planning_data is None else planning_data)
    cells = set().union(*clusters) if clusters else set()
    return policy.update(
        raw_data=raw,
        planning_data=planning,
        width=WIDTH,
        height=HEIGHT,
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        frontier_cells=cells,
        frontier_clusters=list(clusters),
        robot_cell=ROBOT_CELL,
        robot_world=(0.15, 0.55),
        now_s=now_s,
    )


def test_selects_one_target_with_separate_anchor_and_approach():
    policy = make_policy()
    cluster = {(5, column) for column in range(10, 15)}

    update = cycle(policy, [cluster])

    assert update.path[0] == ROBOT_CELL
    assert update.selected_cell == update.path[-1]
    assert policy.target.anchor in cluster
    assert policy.target.attempted_cells == {update.selected_cell}
    assert policy.target.goal_world == (
        update.selected_cell[1] + 0.5,
        update.selected_cell[0] + 0.5,
    )


def test_no_frontier_debounces_and_latches_success():
    policy = make_policy()

    assert cycle(policy, now_s=0.0).debounce_started
    assert not cycle(policy, now_s=7.9).completed_now
    assert cycle(policy, now_s=8.0).completed_now
    assert policy.complete
    assert policy.terminal.outcome == 'success'
    assert cycle(policy, now_s=20.0).path is None


def test_cooldown_defers_completion_then_retries_target():
    policy = make_policy()
    cluster = {(5, column) for column in range(10, 15)}
    policy.memory.register_failure(
        12.5, 5.5, 0.0, 0.75, 30.0, 2, 0.20
    )

    held = cycle(policy, [cluster], now_s=1.0)
    assert held.stats.temporary_rejected == 1
    assert policy.pending_terminal is None

    assigned = cycle(policy, [cluster], now_s=30.0)
    assert assigned.path
    assert policy.target is not None


def test_visited_residual_frontier_is_blocked_after_debounce():
    policy = make_policy()
    cluster = {(5, column) for column in range(10, 15)}
    policy.memory.visited.append((12.5, 5.5))

    cycle(policy, [cluster], now_s=0.0)
    update = cycle(policy, [cluster], now_s=8.0)

    assert update.completed_now
    assert policy.terminal.outcome == 'blocked'
    assert 'visited / too-close' in policy.terminal.blocked_reason


def test_retry_cap_survives_cluster_growth():
    policy = make_policy()
    anchor = (5, 10)
    policy.target = TargetState(
        anchor=anchor,
        goal_world=None,
        attempted_cells={(5, 12)},
        attempted_paths={((5, 1), (5, 12))},
        fresh_approaches=3,
        path_failures=1,
    )
    grown = {(5, column) for column in range(9, 16)}

    update = cycle(policy, [grown])

    assert update.path is None
    assert update.stats.retry_exhausted == 1
    assert policy.target.anchor == anchor
    assert policy.target.fresh_approaches == 3


def test_real_invalidations_preserve_target_and_cap_across_cooldown():
    policy = make_policy(
        maximum_goal_path_failures=1,
        maximum_fresh_approaches_per_target=2,
    )
    cluster = {(5, column) for column in range(10, 17)}

    assigned = scaled_cycle(policy, [cluster], now_s=0.0)
    original_anchor = policy.target.anchor
    assert assigned.path
    assert policy.counters.goals_assigned == 1

    for now_s in (1.0, 2.0):
        grown = cluster | ({(5, 9)} if now_s == 2.0 else set())
        planning = [0] * (WIDTH * HEIGHT)
        goal_x, goal_y = policy.target.goal_world
        goal_cell = int(goal_y / 0.1), int(goal_x / 0.1)
        planning[goal_cell[0] * WIDTH + goal_cell[1]] = 100
        update = scaled_cycle(policy, [grown], now_s, planning)
        assert update.path
        assert policy.target.anchor == original_anchor

    planning = [0] * (WIDTH * HEIGHT)
    goal_x, goal_y = policy.target.goal_world
    goal_cell = int(goal_y / 0.1), int(goal_x / 0.1)
    planning[goal_cell[0] * WIDTH + goal_cell[1]] = 100
    failed = scaled_cycle(policy, [cluster | {(5, 9)}], 3.0, planning)

    assert failed.path is None
    assert policy.counters.goals_assigned == 3
    assert policy.counters.failure_events == 1
    assert policy.target.anchor == original_anchor
    assert policy.target.fresh_approaches == 2
    assert len(policy.target.attempted_cells) == 3
    assert failed.stats.temporary_rejected == 1
    assert policy.pending_terminal is None

    held = scaled_cycle(policy, [cluster | {(5, 9)}], 20.0)
    assert held.path is None
    assert held.stats.temporary_rejected == 1
    assert not policy.complete
    assert policy.pending_terminal is None

    expired = scaled_cycle(policy, [cluster | {(5, 9)}], 33.1)
    assert expired.path is None
    assert expired.stats.retry_exhausted == 1
    assert policy.counters.goals_assigned == 3
    assert policy.pending_terminal is not None


def test_exhausted_target_does_not_block_different_target():
    policy = make_policy()
    exhausted = {(5, column) for column in range(10, 15)}
    different = {(8, column) for column in range(4, 9)}
    policy.target = TargetState(
        anchor=(5, 10),
        goal_world=None,
        fresh_approaches=3,
    )

    update = cycle(policy, [exhausted, different])

    assert update.path
    assert policy.target.anchor in different
    assert policy.target.fresh_approaches == 0


def test_temporary_stuck_recovery_preserves_retry_budget():
    policy = make_policy()
    policy.target = TargetState(
        anchor=(5, 10),
        goal_world=(12.5, 5.5),
        attempted_cells={(5, 12)},
        fresh_approaches=2,
    )

    assert policy.observe_pose(0.0, (1.5, 5.5, 0.0)) is None
    event = policy.observe_pose(5.0, (1.5, 5.5, 0.0))

    assert event.failure_outcome == 'new'
    assert policy.recovery_state == 'requested'
    assert policy.target.anchor == (5, 10)
    assert policy.target.fresh_approaches == 2
    assert policy.target.goal_world is None
    assert policy.observe_pose(6.0, (1.5, 5.5, 0.0)) is None
    assert policy.recovery_status(False)
    assert policy.recovery_state == 'idle'


def test_recovery_request_and_active_status_block_planning_until_inactive():
    policy = make_policy()
    cluster = {(5, column) for column in range(10, 15)}
    policy.recovery_state = 'requested'

    assert cycle(policy, [cluster]).path is None
    assert not policy.recovery_status(True)
    assert cycle(policy, [cluster]).path is None
    assert policy.recovery_status(False)
    assert cycle(policy, [cluster]).path


def test_invalid_path_uses_a_fresh_unattempted_approach():
    policy = make_policy()
    cluster = {(5, column) for column in range(10, 15)}
    original = cycle(policy, [cluster])
    original_cell = original.selected_cell
    policy.target.goal_world = (30.5, 10.5)

    update = cycle(policy, [cluster], now_s=1.0)

    assert update.path
    assert update.selected_cell != original_cell
    assert policy.target.fresh_approaches == 1
    assert original_cell in policy.target.attempted_cells
    assert update.selected_cell in policy.target.attempted_cells


def test_fresh_approach_rejects_an_equivalent_path(monkeypatch):
    policy = make_policy()
    cluster = {(5, column) for column in range(10, 15)}
    duplicate_path = ((5, 1), (5, 2))
    policy.target = TargetState(
        anchor=(5, 10),
        goal_world=(12.5, 5.5),
        attempted_cells={(5, 12)},
        attempted_paths={duplicate_path},
    )
    planning = [0] * (WIDTH * HEIGHT)
    bfs = compute_reachable_component(
        planning, WIDTH, HEIGHT, ROBOT_CELL
    )
    cells = iter(((5, 11), None))
    monkeypatch.setattr(
        'rover_exploration.exploration_policy.'
        'find_reachable_approach',
        lambda **_kwargs: next(cells),
    )
    monkeypatch.setattr(
        'rover_exploration.exploration_policy.reconstruct_grid_path',
        lambda *_args: list(duplicate_path),
    )

    fresh = policy._fresh_approach(
        planning, planning, WIDTH, HEIGHT, 1.0, [cluster], bfs
    )

    assert fresh is None


def test_permanent_failure_abandons_target_and_clears_retry_state():
    policy = make_policy()
    policy.target = TargetState(
        anchor=(5, 10),
        goal_world=(12.5, 5.5),
        attempted_cells={(5, 12)},
        fresh_approaches=2,
    )
    policy.observe_pose(0.0, (1.5, 5.5, 0.0))
    assert policy.observe_pose(5.0, (1.5, 5.5, 0.0))
    policy.recovery_status(False)
    policy.target.goal_world = (12.5, 5.5)

    policy.observe_pose(10.0, (1.5, 5.5, 0.0))
    event = policy.observe_pose(15.0, (1.5, 5.5, 0.0))

    assert event.failure_outcome == 'promoted'
    assert policy.target is None
    assert len(policy.memory.permanent_failures) == 1


def test_reaching_target_clears_retry_state_and_counts_visit():
    policy = make_policy()
    policy.target = TargetState(
        anchor=(5, 10),
        goal_world=ROBOT_WORLD,
        attempted_cells={(5, 10)},
        fresh_approaches=2,
    )

    cycle(policy, [{(5, column) for column in range(10, 15)}])

    assert policy.counters.goals_reached == 1
    assert ROBOT_WORLD in policy.memory.visited
    assert policy.target.fresh_approaches == 0
    assert policy.target.path_failures == 0
    assert len(policy.target.attempted_cells) == 1


def test_result_values_require_and_report_truthful_terminal_state():
    policy = make_policy()
    cycle(policy, now_s=0.0)
    cycle(policy, now_s=8.0)

    result = policy.result_values(8.0)

    assert result['outcome'] == 'success'
    assert result['blocked_reason'] is None
    assert result['geometric_frontier_cells'] == 0
    assert result['post_exclusion_eligible'] == 0
