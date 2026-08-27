from rover_exploration.grid_planning import (
    find_cluster_approach_cell,
)


FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def make_grid(width, height):
    return [FREE] * (width * height)


def test_empty_cluster_returns_none():
    assert find_cluster_approach_cell(
        raw_data=make_grid(3, 3),
        inflated_data=make_grid(3, 3),
        width=3,
        height=3,
        cluster=set(),
    ) is None


def test_traversable_representative_is_preferred():
    # Centroid-nearest member of this cluster is (0, 1).
    cluster = {(0, 0), (0, 1), (1, 1)}

    approach = find_cluster_approach_cell(
        raw_data=make_grid(3, 3),
        inflated_data=make_grid(3, 3),
        width=3,
        height=3,
        cluster=cluster,
    )

    assert approach == (0, 1)


def test_cluster_member_used_when_representative_inflated():
    cluster = {(0, 0), (0, 1), (1, 1)}

    raw_data = make_grid(3, 3)
    inflated_data = make_grid(3, 3)
    inflated_data[0 * 3 + 1] = OCCUPIED
    # (1, 1) is also inflated-blocked, so the nearest safe member
    # by distance to the representative is (0, 0).
    inflated_data[1 * 3 + 1] = OCCUPIED

    approach = find_cluster_approach_cell(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=3,
        height=3,
        cluster=cluster,
    )

    assert approach == (0, 0)


def test_approach_stays_on_reachable_side_of_wall():
    # Every cluster seed is raw-free but inflation-blocked, so the
    # immediate-representative shortcut cannot fire and the BFS must
    # actually run. The only inflated-free approach on the cluster's
    # side of the wall sits behind an inflation ring; a tempting free
    # cell exists across the wall but must never be chosen.
    cluster = {(0, 0), (0, 1), (0, 2)}

    # 5 wide x 4 tall. Row 1 is a solid occupied wall; row 0 is the
    # cluster side, rows 2-3 are open space across the wall.
    width = 5
    height = 4

    raw_data = [FREE] * (width * height)
    for column in range(width):
        raw_data[1 * width + column] = OCCUPIED

    # Inflate the whole wall by one cell: rows 0 and 2 become blocked.
    inflated_data = list(raw_data)
    for column in range(width):
        inflated_data[0 * width + column] = OCCUPIED
        inflated_data[2 * width + column] = OCCUPIED

    # A gap in the wall at (1, 4) is genuinely free on both maps:
    # the reachable approach through it is (0, 4).
    raw_data[1 * width + 4] = FREE
    inflated_data[1 * width + 4] = FREE
    inflated_data[0 * width + 4] = FREE
    inflated_data[2 * width + 4] = FREE

    # Tempting free cell far across the wall.
    inflated_data[3 * width + 1] = FREE

    approach = find_cluster_approach_cell(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=width,
        height=height,
        cluster=cluster,
    )

    # Must stay on the reachable side of the wall (row 0), reached
    # through the wall gap.
    assert approach == (0, 4)


def test_approach_never_crosses_unknown_barrier():
    # Adversarial geometry: every seed is raw-free but
    # inflation-blocked (forcing the BFS branch), a full unknown
    # barrier separates the cluster from another raw-free region,
    # and an inflated-free destination exists ONLY across that
    # barrier. The BFS must refuse to cross unknown cells and
    # return None; if it ever crossed unknown space it would find
    # the tempting destination and this test would fail.
    cluster = {(0, 0), (0, 1), (0, 2)}

    width = 5
    height = 4

    # Row 0: cluster side. Row 1: full unknown barrier.
    # Rows 2-3: raw-free region across the barrier.
    raw_data = [FREE] * (width * height)

    for column in range(width):
        raw_data[1 * width + column] = UNKNOWN

    # Inflate one cell onto rows 0 and 2 so no inflated-free cell
    # exists on the cluster's reachable side (row 0).
    inflated_data = list(raw_data)

    for column in range(width):
        if inflated_data[0 * width + column] == FREE:
            inflated_data[0 * width + column] = OCCUPIED

        if inflated_data[2 * width + column] != UNKNOWN:
            inflated_data[2 * width + column] = OCCUPIED

    # Tempting inflated-free destinations across the barrier.
    inflated_data[3 * width + 1] = FREE
    inflated_data[3 * width + 3] = FREE

    approach = find_cluster_approach_cell(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=width,
        height=height,
        cluster=cluster,
    )

    # Correct result: None, because crossing unknown is forbidden
    # and no safe approach exists on the reachable side.
    assert approach is None


def test_none_when_no_raw_free_approach_exists():
    cluster = {(0, 0), (0, 1)}

    data = [OCCUPIED] * 6

    assert find_cluster_approach_cell(
        raw_data=data,
        inflated_data=data,
        width=3,
        height=2,
        cluster=cluster,
    ) is None


def test_unknown_cells_are_never_crossed():
    cluster = {(0, 0)}

    raw_data = make_grid(5, 1)
    raw_data[0] = UNKNOWN

    approach = find_cluster_approach_cell(
        raw_data=raw_data,
        inflated_data=raw_data,
        width=5,
        height=1,
        cluster=cluster,
    )

    assert approach is None
