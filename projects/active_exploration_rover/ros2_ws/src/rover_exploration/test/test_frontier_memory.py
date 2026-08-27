# Tests for lifetime failure records vs. active cooldowns.
#
# Key invariant: cooldown pruning must never erase the lifetime
# failure count, so a second failure — even long after the first
# cooldown expired — still promotes the region to permanent.

from rover_exploration.frontier_memory import (
    is_excluded,
    prune_expired_cooldowns,
    record_failure,
)
from rover_exploration.grid_planning import (
    compute_reachable_component,
    select_cluster_weighted_goal,
)

FREE = 0
OCCUPIED = 100


def make_memory():
    return {
        'records': [],
        'permanent': [],
        'visited': [],
    }


def fail(memory, x, y, now_s):
    return record_failure(
        failure_records=memory['records'],
        permanent_regions=memory['permanent'],
        x=x, y=y, now_s=now_s,
        match_radius_m=0.75,
        blacklist_duration_s=30.0,
        promotion_failures=2,
    )


def excluded(memory, x, y, now_s):
    return is_excluded(
        x=x, y=y,
        failure_records=memory['records'],
        permanent_regions=memory['permanent'],
        visited_regions=memory['visited'],
        now_s=now_s,
        exclusion_radius_m=0.75,
        visited_radius_m=0.60,
    )


def test_second_failure_before_temporary_expiry():
    memory = make_memory()

    assert fail(memory, 1.0, 1.0, 0.0) == 'new'
    assert excluded(memory, 1.0, 1.0, 10.0) == 'temporary'

    outcome = fail(memory, 1.05, 1.02, 20.0)

    assert outcome == 'promoted'
    assert memory['permanent'] == [(1.0, 1.0)]
    assert excluded(memory, 1.0, 1.0, 20.0) == 'permanent'


def test_second_failure_after_temporary_expiry_promotes():
    """The core regression: expiry must not erase the first count."""
    memory = make_memory()

    # t=0: goal A fails; temporarily excluded.
    assert fail(memory, 5.0, 5.0, 0.0) == 'new'
    assert excluded(memory, 5.0, 5.0, 10.0) == 'temporary'

    # t=31: prune expired cooldowns; A becomes eligible again.
    prune_expired_cooldowns(
        failure_records=memory['records'], now_s=31.0
    )
    assert excluded(memory, 5.0, 5.0, 31.0) is None

    # t=40: a second failure near A must see the old evidence.
    outcome = fail(memory, 5.1, 4.95, 40.0)

    assert outcome == 'promoted'
    assert memory['permanent'] == [(5.0, 5.0)]
    assert memory['records'][0]['failure_count'] == 2


def test_spatially_shifted_failure_after_expiry():
    memory = make_memory()

    assert fail(memory, 8.0, 8.0, 0.0) == 'new'

    prune_expired_cooldowns(
        failure_records=memory['records'], now_s=100.0
    )
    assert excluded(memory, 8.0, 8.0, 100.0) is None

    # SLAM drifts the frontier ~0.45 m away: same spatial region.
    outcome = fail(memory, 8.3, 8.35, 120.0)

    assert outcome == 'promoted'
    assert memory['permanent'] == [(8.0, 8.0)]


def test_permanent_exclusion_survives_arbitrarily_later_times():
    memory = make_memory()

    fail(memory, 3.0, 7.0, 0.0)
    fail(memory, 3.1, 6.95, 50.0)

    assert memory['permanent'] == [(3.0, 7.0)]

    for now_s in (60.0, 500.0, 10_000.0, 1e9):
        assert (
            excluded(memory, 3.05, 7.0, now_s)
            == 'permanent'
        )


def test_new_failure_still_gets_a_cooldown():
    memory = make_memory()

    assert fail(memory, 1.0, 1.0, 0.0) == 'new'
    assert memory['records'][0]['blocked_until_s'] == 30.0
    assert excluded(memory, 1.0, 1.0, 29.9) == 'temporary'
    assert excluded(memory, 1.0, 1.0, 30.1) is None


def test_pruning_never_deletes_lifetime_records():
    memory = make_memory()

    fail(memory, 2.0, 2.0, 0.0)

    for tick in (31.0, 62.0, 93.0):
        prune_expired_cooldowns(
            failure_records=memory['records'], now_s=tick
        )

    assert len(memory['records']) == 1
    assert memory['records'][0]['failure_count'] == 1
    assert not memory['permanent']


# --- Cluster-weighted selection ---

def make_bfs(width, height, start=(1, 1)):
    data = [FREE] * (width * height)
    return compute_reachable_component(
        data=data, width=width, height=height,
        start=start,
    )


def test_larger_cluster_wins_within_distance_slack():
    bfs = make_bfs(21, 3)

    candidates = {
        (1, 5): (2,),    # small cluster, cost 4
        (1, 12): (8,),   # large cluster, cost 11
    }

    result = select_cluster_weighted_goal(
        bfs=bfs,
        candidate_costs=candidates,
        distance_slack_cells=40,
    )

    assert result is not None
    assert result[0] == (1, 12)


def test_nearest_wins_when_large_cluster_outside_slack():
    bfs = make_bfs(101, 3)

    # Shortest cost is 4. The huge cluster sits at cost 40: far
    # beyond shortest_cost + slack(3), so it is not shortlisted.
    candidates = {
        (1, 5): (2,),     # small cluster, near
        (1, 41): (50,),   # huge cluster, cost 40
    }

    result = select_cluster_weighted_goal(
        bfs=bfs,
        candidate_costs=candidates,
        distance_slack_cells=3,
    )

    assert result is not None
    assert result[0] == (1, 5)


def test_unreachable_candidates_are_ignored():
    width = 11
    height = 3
    data = [FREE] * (width * height)

    for row in range(height):
        data[row * width + 5] = OCCUPIED

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(1, 1),
    )

    result = select_cluster_weighted_goal(
        bfs=bfs,
        candidate_costs={(1, 7): (9,)},
        distance_slack_cells=40,
    )

    assert result is None


def test_selection_is_deterministic():
    results = set()

    for _ in range(5):
        bfs = make_bfs(21, 3)
        result = select_cluster_weighted_goal(
            bfs=bfs,
            candidate_costs={
                (1, 8): (5,),
                (1, 14): (5,),
            },
            distance_slack_cells=40,
        )
        results.add(result[0])

    assert len(results) == 1


def test_overlapping_cooldowns_any_active_excludes():
    """Expired record must not mask an active overlapping one."""
    memory = make_memory()

    # Record A: expired long ago, centred at x=0.
    memory['records'].append({
        'x': 0.0,
        'y': 0.0,
        'failure_count': 1,
        'blocked_until_s': 10.0,
    })

    # Record B: active cooldown, centred at x=1.4.
    memory['records'].append({
        'x': 1.4,
        'y': 0.0,
        'failure_count': 1,
        'blocked_until_s': 5000.0,
    })

    # Candidate at x=0.7 is within 0.75 m of both records.
    exclusion = excluded(memory, 0.7, 0.0, 50.0)

    assert exclusion == 'temporary'
