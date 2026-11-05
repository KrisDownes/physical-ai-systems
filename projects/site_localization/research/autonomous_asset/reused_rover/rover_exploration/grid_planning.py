from collections import deque

neighbor_offsets = (
    (-1, 0),  # north
    (1, 0),   # south
    (0, -1),  # west
    (0, 1),   # east
    )


def _disk_offsets(radius):
    """Structuring element offsets for the closing radius."""
    # Radius 1 uses the full 3x3 square (8-connected): a
    # plus-shaped element cannot bridge a pinhole whose vertical
    # neighbours are free, but the square element does. Larger
    # radii remain full Euclidean disks.
    if radius == 1:
        return [
            (row_offset, column_offset)
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
        ]

    offsets = []

    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            distance_squared = (
                row_offset ** 2 + column_offset ** 2
            )

            if distance_squared <= radius ** 2:
                offsets.append((row_offset, column_offset))

    return offsets


def close_occupied_walls(
    data,
    width,
    height,
    closing_radius_cells=1,
) -> list[int]:
    # Genuine binary closing of the occupied mask.
    #
    # Morphological closing: ``closed = dilate(erode(dilate(mask)))``
    # with a disk structuring element of ``closing_radius_cells``.
    # Only originally free cells may change value (to blocked), and
    # only where the closing fills a gap narrower than the
    # structuring element. Guarantees:
    #
    # * every originally occupied cell stays occupied;
    # * unknown cells keep their raw -1 and stay non-traversable;
    # * a one/two-cell pinhole is sealed because its cells have no
    #   eroded core to grow back from;
    # * a doorway wider than twice the radius keeps traversable free
    #   cells and survives;
    # * the input list is never mutated.
    if width <= 0 or height <= 0:
        raise ValueError('Dimensions must be positive')

    if len(data) != width * height:
        raise ValueError('Invalid measurement amount')

    if closing_radius_cells < 0:
        raise ValueError(
            'Closing radius must be non-negative'
        )

    result = list(data)

    if closing_radius_cells == 0:
        return result

    offsets = _disk_offsets(closing_radius_cells)

    def index_of(row, column):
        return row * width + column

    def dilate(mask):
        grown = list(mask)

        for row in range(height):
            for column in range(width):
                if not mask[index_of(row, column)]:
                    continue

                for row_offset, column_offset in offsets:
                    neighbor_row = (
                        row + row_offset
                    )
                    neighbor_column = (
                        column + column_offset
                    )

                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    ):
                        grown[
                            index_of(
                                neighbor_row, neighbor_column
                            )
                        ] = True

        return grown

    def erode(mask):
        # Cells outside the grid count as mask, so erosion does not
        # eat wall cores that touch the boundary.
        shrunk = [False] * (width * height)

        for row in range(height):
            for column in range(width):
                fits = True

                for row_offset, column_offset in offsets:
                    neighbor_row = (
                        row + row_offset
                    )
                    neighbor_column = (
                        column + column_offset
                    )

                    if not (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    ):
                        # Outside the grid counts as occupied:
                        # a wall ending at the border keeps its
                        # full thickness through erosion.
                        continue

                    if not mask[
                        index_of(neighbor_row, neighbor_column)
                    ]:
                        fits = False
                        break

                shrunk[index_of(row, column)] = fits

        return shrunk

    occupied_mask = [
        value == 100 for value in data
    ]

    # True morphological closing: erode(dilate(mask)). The final
    # dilation restores the original mask's outer boundary, so an
    # isolated obstacle keeps its exact footprint (no cross-growth),
    # while a gap narrower than the structuring element has no
    # eroded core and stays sealed.
    closed_mask = erode(dilate(occupied_mask))

    for index in range(width * height):
        if (
            data[index] == 0
            and closed_mask[index]
        ):
            result[index] = 100

    return result


def build_planning_grid(
    raw_data,
    inflated_data,
    conditioned_data,
):
    # Merge all blocking sources into the final planning grid.
    # A cell is planning-blocked (100) when ANY source blocks it:
    # occupied inflation, wall closing, or unknown padding. Raw
    # unknown cells (-1) are preserved exactly as -1 so they remain
    # explicitly non-traversable and visually distinct in RViz --
    # they must never silently become traversable 0. The input
    # lists are never mutated.

    if not (
        len(raw_data)
        == len(inflated_data)
        == len(conditioned_data)
    ):
        raise ValueError(
            'Planning grid inputs must have equal lengths'
        )

    planning_data = []

    for raw, inflated, conditioned in zip(
        raw_data, inflated_data, conditioned_data
    ):
        # Raw unknown keeps its -1 identity regardless of derived
        # blocking: it is never silently rewritten to 0 or 100.
        if raw == -1:
            planning_data.append(-1)
        elif inflated == 100 or conditioned == 100:
            planning_data.append(100)
        else:
            planning_data.append(0)

    return planning_data


def pad_unknown_space(
    data,
    width,
    height,
    padding_radius_cells=2,
) -> list[int]:
    # Only raw-free cells absorb the padding; raw unknown cells stay
    # exactly where they are and remain non-traversable. This is an
    # independent, deliberately small buffer around unknown space —
    # it must never receive the full seven-cell occupied inflation.
    if width <= 0 or height <= 0:
        raise ValueError('Dimensions must be positive')

    if len(data) != width * height:
        raise ValueError('Invalid measurement amount')

    if padding_radius_cells < 0:
        raise ValueError(
            'Padding radius must be non-negative'
        )

    result = list(data)

    if padding_radius_cells == 0:
        return result

    offsets = _disk_offsets(padding_radius_cells)

    def index_of(row, column):
        return row * width + column

    padded_blocked = set()

    for row in range(height):
        for column in range(width):
            if data[index_of(row, column)] != -1:
                continue

            for row_offset, column_offset in offsets:
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset

                if not (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                ):
                    continue

                if (
                    data[
                        index_of(neighbor_row, neighbor_column)
                    ]
                    == 0
                ):
                    padded_blocked.add(
                        index_of(neighbor_row, neighbor_column)
                    )

    for index in padded_blocked:
        result[index] = 100

    return result


def is_traversable_grid_cell(
    data,
    width,
    height,
    row,
    column,
) -> bool:

    if width <= 0 or height <= 0:
        raise ValueError('Dimensions must be positive')
    if len(data) != width * height:
        raise ValueError('Invalid measurement amount')

    if not (
        0 <= row < height
        and 0 <= column < width
    ):
        return False

    index = row * width + column

    return data[index] == 0


def traversable_grid_neighbors(
    data,
    width,
    height,
    cell,
) -> list[tuple[int, int]]:

    traversable_neighbors = []

    row, column = cell

    for row_offset, column_offset in neighbor_offsets:
        neighbor_row = row + row_offset
        neighbor_column = column + column_offset

        if is_traversable_grid_cell(
            data=data,
            width=width,
            height=height,
            row=neighbor_row,
            column=neighbor_column,
        ):
            traversable_neighbors.append(
                (neighbor_row, neighbor_column)
            )

    return traversable_neighbors


def manhattan_grid_distance(
    first_cell,
    second_cell,
) -> int:

    first_row, first_column = first_cell
    second_row, second_column = second_cell

    return (
        abs(first_row - second_row)
        + abs(first_column - second_column)
    )


def reconstruct_grid_path(
    came_from,
    current_cell,
) -> list[tuple[int, int]]:

    trail = [current_cell]
    while current_cell in came_from:
        current_cell = came_from[current_cell]
        trail.append(current_cell)
    trail.reverse()

    return trail


def find_escape_path(
    raw_data,
    inflated_data,
    width,
    height,
    start,
) -> list[tuple[int, int]] | None:
    # Walk only through cells that are free (0) on the raw occupancy
    # grid, so the search can never cross a real wall or unknown
    # space. Stop at the first cell that is also free on the inflated
    # grid: the nearest cell where normal planning is allowed.

    if start is None:
        return None

    if not is_traversable_grid_cell(
        data=raw_data,
        width=width,
        height=height,
        row=start[0],
        column=start[1],
    ):
        return None

    if is_traversable_grid_cell(
        data=inflated_data,
        width=width,
        height=height,
        row=start[0],
        column=start[1],
    ):
        return [start]

    came_from = {}
    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()

        if is_traversable_grid_cell(
            data=inflated_data,
            width=width,
            height=height,
            row=current[0],
            column=current[1],
        ):
            return reconstruct_grid_path(came_from, current)

        for neighbor in traversable_grid_neighbors(
            data=raw_data,
            width=width,
            height=height,
            cell=current,
        ):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            came_from[neighbor] = current
            queue.append(neighbor)

    return None


def inflate_occupancy_grid(
    data,
    width,
    height,
    inflation_radius_cells,
) -> list[int]:

    if inflation_radius_cells < 0:
        raise ValueError('Inflation radius must be non-negative')

    if width <= 0 or height <= 0:
        raise ValueError('Dimensions must be positive')

    if len(data) != width * height:
        raise ValueError('Invalid measurement amount')

    inflated_data = list(data)

    inflation_offsets = []

    for row_offset in range(
        -inflation_radius_cells,
        inflation_radius_cells + 1,
    ):
        for column_offset in range(
            -inflation_radius_cells,
            inflation_radius_cells + 1,
        ):
            distance_squared = (
                row_offset ** 2
                + column_offset ** 2
            )

            if distance_squared <= inflation_radius_cells ** 2:
                inflation_offsets.append(
                    (row_offset, column_offset)
                )

    for row in range(height):
        for column in range(width):
            index = row * width + column
            if data[index] <= 0:
                continue

            for row_offset, column_offset in inflation_offsets:
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset

                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                ):
                    neighbor_index = (
                        neighbor_row * width + neighbor_column
                    )

                    if inflated_data[neighbor_index] == 0:
                        inflated_data[neighbor_index] = 100

    return inflated_data


def compute_reachable_component(
    data,
    width,
    height,
    start,
) -> dict | None:
    # One BFS from the rover over the planning grid. Returns a
    # dict with three consistent views of the same search:
    #
    # * reachable: boolean grid, True for every cell in the
    #   rover's component.
    # * cost: route distance in cells per reachable cell.
    # * came_from: parent map for path reconstruction.
    #
    # Returns None when the start cell itself is not traversable --
    # callers must fail closed. Computed once per map cycle;
    # candidate filtering, cluster approach search, goal selection,
    # and path reconstruction all share this single result.

    if start is None:
        return None

    if not is_traversable_grid_cell(
        data=data,
        width=width,
        height=height,
        row=start[0],
        column=start[1],
    ):
        return None

    reachable = [False] * (width * height)
    reachable[start[0] * width + start[1]] = True

    cost = {start: 0}
    came_from = {}
    queue = deque([start])

    while queue:
        current = queue.popleft()

        for neighbor in traversable_grid_neighbors(
            data=data,
            width=width,
            height=height,
            cell=current,
        ):
            index = (
                neighbor[0] * width + neighbor[1]
            )

            if reachable[index]:
                continue

            reachable[index] = True
            cost[neighbor] = cost[current] + 1
            came_from[neighbor] = current
            queue.append(neighbor)

    return {
        'reachable': reachable,
        'cost': cost,
        'came_from': came_from,
        'start': start,
        'width': width,
    }
