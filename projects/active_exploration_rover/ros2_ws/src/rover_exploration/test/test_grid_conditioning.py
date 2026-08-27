"""Tests for the derived planning-grid conditioning helpers."""

from rover_exploration.grid_planning import (
    build_planning_grid,
    close_occupied_walls,
    find_escape_path,
    find_grid_path,
    pad_unknown_space,
)


FREE = 0
OCCUPIED = 100
UNKNOWN = -1


def make_wall_grid(width, height, wall_row, door_columns):
    """Solid horizontal occupied wall with optional door cells."""
    data = [FREE] * (width * height)

    for column in range(width):
        data[wall_row * width + column] = OCCUPIED

    for column in door_columns:
        data[wall_row * width + column] = FREE

    return data


def test_one_cell_pinhole_is_sealed():
    width = 9
    height = 9
    data = make_wall_grid(width, height, 4, [])

    # Drill a single-cell pinhole in the middle of the wall.
    data[4 * width + 4] = FREE

    conditioned = close_occupied_walls(
        data, width, height, closing_radius_cells=1
    )

    assert (
        conditioned[4 * width + 4] == OCCUPIED
    )


def test_real_doorway_stays_open():
    width = 13
    height = 13
    door_center = 6
    data = make_wall_grid(
        width, height, 4, [door_center - 1,
                           door_center, door_center + 1]
    )

    conditioned = close_occupied_walls(
        data, width, height, closing_radius_cells=1
    )

    # The doorway interior keeps at least one traversable cell
    # with free vertical connectivity through the wall.
    open_cells = [
        column
        for column in (door_center - 1, door_center,
                       door_center + 1)
        if conditioned[4 * width + column] == FREE
    ]

    assert open_cells, 'doorway fully sealed'

    for column in open_cells:
        assert conditioned[
            3 * width + column
        ] == FREE
        assert conditioned[
            5 * width + column
        ] == FREE


def test_unknown_barrier_remains_uncrossable():
    width = 5
    height = 3

    raw_data = [FREE] * (width * height)

    for column in range(width):
        raw_data[1 * width + column] = UNKNOWN

    padded = pad_unknown_space(
        raw_data, width, height, padding_radius_cells=2
    )

    # The padded grid blocks every row-0 and row-2 cell.
    for column in range(width):
        assert padded[
            0 * width + column
        ] == OCCUPIED or raw_data[
            0 * width + column
        ] != FREE
        assert padded[2 * width + column] == OCCUPIED

    start = (0, 0)
    goal = (2, 2)

    padded[start[0] * width + start[1]] = FREE
    padded[goal[0] * width + goal[1]] = FREE

    path = find_grid_path(
        data=padded,
        width=width,
        height=height,
        start=start,
        goal=goal,
    )

    assert path is None


def test_unknown_padding_does_not_get_full_inflation():
    # A single unknown cell must block only its small padding
    # radius, not a seven-cell obstacle-inflation ring.
    width = 21
    height = 21
    center = 10 * width + 10

    raw_data = [FREE] * (width * height)
    raw_data[center] = UNKNOWN

    padded = pad_unknown_space(
        raw_data, width, height, padding_radius_cells=2
    )

    blocked = [
        index for index in range(width * height)
        if padded[index] == OCCUPIED
        and raw_data[index] == FREE
    ]

    # All blocked cells lie within the 2-cell padding disk.
    for index in blocked:
        row = index // width
        column = index % width

        distance_squared = (
            (row - 10) ** 2 + (column - 10) ** 2
        )

        assert distance_squared <= 2 ** 2 + 1

    # Cells beyond the padding ring stay free: no seven-cell
    # inflation leaked into the unknown padding.
    far_index = 10 * width + 16
    assert padded[far_index] == FREE


def test_conditioning_helpers_do_not_mutate_input():
    width = 7
    height = 7

    raw_data = make_wall_grid(width, height, 3, [])
    raw_data[3 * width + 3] = UNKNOWN
    original = list(raw_data)

    close_occupied_walls(
        raw_data, width, height, closing_radius_cells=1
    )
    assert raw_data == original

    pad_unknown_space(
        raw_data, width, height, padding_radius_cells=2
    )
    assert raw_data == original


def test_escape_path_through_derived_padding():
    # A rover sitting inside unknown-derived padding can still get
    # a raw-free escape corridor to planning-safe space.
    width = 11
    height = 1

    raw_data = [FREE] * width
    inflated_data = [FREE] * width

    # Unknown cell at column 0; its 2-cell padding blocks the
    # first three columns of the planning grid.
    raw_data[0] = UNKNOWN
    inflated_data[0] = OCCUPIED
    inflated_data[1] = OCCUPIED
    inflated_data[2] = OCCUPIED

    path = find_escape_path(
        raw_data=raw_data,
        inflated_data=inflated_data,
        width=width,
        height=height,
        start=(0, 1),
    )

    assert path is not None
    assert path[0] == (0, 1)
    assert path[-1] == (0, 3)

    for row, column in path:
        assert raw_data[row * width + column] == FREE


def test_planned_path_cannot_cross_sealed_wall_or_unknown():
    width = 9
    height = 9

    raw_data = make_wall_grid(width, height, 4, [])
    raw_data[4 * width + 4] = FREE  # pinhole

    # Raw unknown pocket on the far side of the sealed pinhole.
    raw_data[6 * width + 4] = UNKNOWN

    planning_data = close_occupied_walls(
        raw_data, width, height, closing_radius_cells=1
    )
    planning_data = pad_unknown_space(
        planning_data, width, height,
        padding_radius_cells=2,
    )

    start = (0, 4)
    goal = (8, 4)

    path = find_grid_path(
        data=planning_data,
        width=width,
        height=height,
        start=start,
        goal=goal,
    )

    assert path is None


# --- build_planning_grid merge (v6 blocker 1) ---

def test_build_planning_grid_preserves_raw_unknown():
    raw = [FREE, UNKNOWN, FREE]
    inflated = [FREE, FREE, OCCUPIED]
    conditioned = [FREE, OCCUPIED, FREE]

    merged = build_planning_grid(
        raw_data=raw,
        inflated_data=inflated,
        conditioned_data=conditioned,
    )

    # Raw unknown stays -1 even though conditioning blocked it:
    # unknown must never silently become traversable 0.
    assert merged == [0, -1, 100]


def test_build_planning_grid_merges_all_blockers():
    raw = [FREE] * 4
    inflated = [FREE, OCCUPIED, FREE, FREE]
    conditioned = [OCCUPIED, FREE, FREE, FREE]

    merged = build_planning_grid(
        raw_data=raw,
        inflated_data=inflated,
        conditioned_data=conditioned,
    )

    assert merged == [100, 100, 0, 0]


def test_build_planning_grid_rejects_unequal_lengths():
    try:
        build_planning_grid(
            raw_data=[FREE],
            inflated_data=[FREE, FREE],
            conditioned_data=[FREE],
        )
        raised = False
    except ValueError:
        raised = True

    assert raised is True


def test_isolated_occupied_cell_is_not_grown():
    # v6 regression: dilation-only "closing" expanded one occupied
    # cell into a five-cell cross. True erode(dilate()) closing
    # must leave a lone obstacle exactly as it was.
    data = [FREE] * 49
    data[24] = OCCUPIED

    conditioned = close_occupied_walls(
        data, 7, 7, closing_radius_cells=1
    )

    assert sum(
        1 for value in conditioned if value == OCCUPIED
    ) == 1
