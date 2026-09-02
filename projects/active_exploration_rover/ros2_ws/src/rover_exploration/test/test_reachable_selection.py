"""Reachable approach selection and completion debounce tests."""

import pytest

from rover_exploration.frontier_selection import (
    CandidateSet,
    find_reachable_approach,
    select_weighted_goal,
)
from rover_exploration.grid_planning import compute_reachable_component

FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def make_bfs(width, height, start=(1, 1), data=None):
    if data is None:
        data = [FREE] * (width * height)

    return compute_reachable_component(
        data=data, width=width, height=height,
        start=start,
    )


# --- Core regression: disconnected first choice must fall back ---

def test_disconnected_first_choice_falls_back_to_reachable():
    # Realistic disconnected-first-choice geometry. The frontier
    # cluster is a connected raw-free blob entirely inside the
    # rover's own room (rows 0-2), so the real clustering algorithm
    # could actually produce it. Its representative (1, 2) is
    # raw-free but planning-blocked by padding, so it is NOT a valid
    # standoff. The search must traverse outward (still raw-free) to
    # (0, 2): the first planning-free, reachable standoff cell. The
    # far room is sealed by a solid wall and is never part of this
    # cluster -- that straddling geometry is what the old test
    # smuggled in and the real cluster maker cannot emit.

    width = 11
    height = 7

    data = [FREE] * (width * height)

    # Solid occupied wall across row 3 with NO doorway.
    for column in range(width):
        data[3 * width + column] = OCCUPIED

    # Rover starts on row 0.
    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(0, 1),
    )

    reachable = bfs['reachable']

    # Connected near-side cluster; all of its cells plus the cells
    # immediately below it are planning-blocked so the only valid
    # standoff lies above the cluster.
    cluster = {
        (1, 2),
        (1, 3),
    }

    planning_data = list(data)
    planning_data[1 * width + 2] = UNKNOWN
    planning_data[1 * width + 3] = OCCUPIED
    planning_data[2 * width + 2] = OCCUPIED
    planning_data[2 * width + 3] = OCCUPIED

    approach = find_reachable_approach(
        raw_data=data,
        planning_data=planning_data,
        width=width,
        height=height,
        cluster=cluster,
        bfs=bfs,
        max_search_radius_cells=8,
    )

    assert approach == (0, 2)

    # Standoff is reachable and planning-free.
    assert reachable[0 * width + 2] is True
    assert planning_data[0 * width + 2] == FREE


def test_unreachable_cluster_returns_none_within_bound():
    """A fully disconnected cluster maps to no approach cell."""
    width = 11
    height = 5

    data = [FREE] * (width * height)

    # Solid wall, no doorway.
    for column in range(width):
        data[2 * width + column] = OCCUPIED

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(0, 1),
    )

    cluster = {(4, 5), (4, 6)}

    approach = find_reachable_approach(
        raw_data=data,
        planning_data=data,
        width=width,
        height=height,
        cluster=cluster,
        bfs=bfs,
        max_search_radius_cells=3,
    )

    assert approach is None


def test_approach_search_radius_bounds_expansion():
    """A planning-blocked seed whose only standoff is beyond the short limit."""
    width = 21
    height = 3

    data = [FREE] * (width * height)

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(1, 1),
    )

    # Seed is raw-free but planning-blocked. A five-cell-wide
    # planning-blocked plug sits over the seed; the nearest
    # planning-free raw-free cell is three cells away along the row.
    cluster = {(1, 10)}

    planning_data = list(data)
    for column in (8, 9, 10, 11, 12):
        for row in range(height):
            planning_data[row * width + column] = OCCUPIED

    limited = find_reachable_approach(
        raw_data=data,
        planning_data=planning_data,
        width=width,
        height=height,
        cluster=cluster,
        bfs=bfs,
        max_search_radius_cells=2,
    )

    assert limited is None

    within = find_reachable_approach(
        raw_data=data,
        planning_data=planning_data,
        width=width,
        height=height,
        cluster=cluster,
        bfs=bfs,
        max_search_radius_cells=5,
    )

    assert within is not None
    # Reached a free cell strictly beyond the short (2-cell) limit.
    assert abs(within[1] - 10) > 2


def test_raw_occupied_barrier_cannot_be_crossed():
    """The approach search never traverses a raw occupied cell."""
    width = 11
    height = 3

    data = [FREE] * (width * height)
    # Vertical occupied wall at column 5 with no doorway.
    for row in range(height):
        data[row * width + 5] = OCCUPIED

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(1, 1),
    )

    # The entire near side is planning-blocked, so the only possible
    # standoff would be across the raw occupied wall -- unreachable
    # and non-traversable. The search must return None.
    cluster = {(1, 3)}

    planning_data = list(data)
    for row in range(height):
        for column in range(5):
            planning_data[row * width + column] = OCCUPIED

    approach = find_reachable_approach(
        raw_data=data,
        planning_data=planning_data,
        width=width,
        height=height,
        cluster=cluster,
        bfs=bfs,
        max_search_radius_cells=12,
    )

    assert approach is None


def test_raw_unknown_barrier_cannot_be_crossed():
    """The approach search never traverses a raw unknown cell."""
    width = 11
    height = 3

    data = [FREE] * (width * height)
    # Vertical unknown barrier at column 5 (never classified free).
    for row in range(height):
        data[row * width + 5] = UNKNOWN

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(1, 1),
    )

    cluster = {(1, 3)}

    planning_data = list(data)
    for row in range(height):
        for column in range(5):
            planning_data[row * width + column] = OCCUPIED

    approach = find_reachable_approach(
        raw_data=data,
        planning_data=planning_data,
        width=width,
        height=height,
        cluster=cluster,
        bfs=bfs,
        max_search_radius_cells=12,
    )

    assert approach is None


# --- Shared BFS tree for selection and path reconstruction ---


def test_selector_reuses_bfs_tree_without_second_search():
    width = 11
    height = 3
    data = [FREE] * (width * height)

    for row in range(height):
        data[row * width + 5] = OCCUPIED

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(1, 1),
    )

    result = select_weighted_goal(
        bfs=bfs,
        candidates=CandidateSet({(1, 8): 9}, {}, 0, 0),
        eligible=[(1, 8)],
        distance_slack_cells=40,
    )

    assert result is None


def test_selector_path_is_contiguous_and_reachable():
    width = 21
    height = 3
    data = [FREE] * (width * height)

    bfs = compute_reachable_component(
        data=data, width=width, height=height,
        start=(1, 1),
    )

    result = select_weighted_goal(
        bfs=bfs,
        candidates=CandidateSet({(1, 15): 4}, {}, 0, 0),
        eligible=[(1, 15)],
        distance_slack_cells=40,
    )

    assert result is not None
    candidate, path = result

    assert path[0] == (1, 1)
    assert path[-1] == candidate

    for first, second in zip(path, path[1:]):
        delta = (
            abs(first[0] - second[0])
            + abs(first[1] - second[1])
        )
        assert delta == 1


@pytest.mark.parametrize(
    'sizes,costs,slack,expected',
    (
        ({(1, 4): 2, (1, 6): 8}, {(1, 4): 3, (1, 6): 5}, 2, (1, 6)),
        ({(1, 4): 2, (1, 8): 8}, {(1, 4): 3, (1, 8): 7}, 2, (1, 4)),
        ({(1, 4): 5, (2, 3): 5}, {(1, 4): 3, (2, 3): 3}, 0, (1, 4)),
    ),
)
def test_weighted_selection_is_bounded_and_deterministic(
    sizes, costs, slack, expected
):
    bfs = {
        'cost': costs,
        'came_from': {cell: None for cell in costs},
    }
    candidates = CandidateSet(sizes, {}, 0, 0)

    selected, _path = select_weighted_goal(
        bfs, candidates, list(sizes), slack
    )

    assert selected == expected
