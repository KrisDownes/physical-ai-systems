from collections import deque
import heapq

from rover_exploration.frontier_detection import (
    representative_frontier_cell,
)

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


def find_cluster_approach_cell(
    raw_data,
    inflated_data,
    width,
    height,
    cluster,
) -> tuple[int, int] | None:
    # The cluster cells are the search seeds. The search walks only
    # through cells free on the raw grid, so it cannot cross real walls
    # or unknown space, and selects the nearest cell that is also free
    # on the inflated grid. Returns None when no raw-free approach to
    # the cluster exists.

    if not cluster:
        return None

    representative = representative_frontier_cell(cluster)

    if is_traversable_grid_cell(
        data=inflated_data,
        width=width,
        height=height,
        row=representative[0],
        column=representative[1],
    ):
        return representative

    queue = deque()
    visited = set()

    # Prefer traversable cluster members near the representative
    # before expanding into the surrounding raw-free region.
    for cell in sorted(
        cluster,
        key=lambda member: manhattan_grid_distance(
            member, representative
        ),
    ):
        if is_traversable_grid_cell(
            data=raw_data,
            width=width,
            height=height,
            row=cell[0],
            column=cell[1],
        ) and is_traversable_grid_cell(
            data=inflated_data,
            width=width,
            height=height,
            row=cell[0],
            column=cell[1],
        ):
            return cell

        if is_traversable_grid_cell(
            data=raw_data,
            width=width,
            height=height,
            row=cell[0],
            column=cell[1],
        ):
            queue.append(cell)
            visited.add(cell)

    while queue:
        current = queue.popleft()

        if is_traversable_grid_cell(
            data=inflated_data,
            width=width,
            height=height,
            row=current[0],
            column=current[1],
        ):
            return current

        for neighbor in traversable_grid_neighbors(
            data=raw_data,
            width=width,
            height=height,
            cell=current,
        ):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append(neighbor)

    return None


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


def find_grid_path(
    data,
    width,
    height,
    start,
    goal,
) -> list[tuple[int, int]] | None:

    if start is None or goal is None:
        return None

    if not is_traversable_grid_cell(
        data=data,
        width=width,
        height=height,
        row=start[0],
        column=start[1],
    ):
        return None

    if not is_traversable_grid_cell(
        data=data,
        width=width,
        height=height,
        row=goal[0],
        column=goal[1],
    ):
        return None

    came_from = {}
    g_score = {start: 0}
    visited = set()
    open_heap = []

    start_h = manhattan_grid_distance(start, goal)

    heapq.heappush(
        open_heap,
        (
            start_h,
            start_h,
            start[0],
            start[1],
        ),
        )

    while open_heap:
        _, _, current_row, current_column = (
            heapq.heappop(open_heap)
        )

        current = (current_row, current_column)

        if current in visited:
            continue

        if current == goal:
            return reconstruct_grid_path(
                came_from,
                current
            )

        visited.add(current)

        for neighbor in traversable_grid_neighbors(
            data=data,
            width=width,
            height=height,
            cell=current,
        ):

            if neighbor in visited:
                continue

            tentative_g = g_score[current] + 1

            known_g = g_score.get(
                neighbor,
                float('inf'),
            )

            if tentative_g < known_g:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                neighbor_h = manhattan_grid_distance(neighbor, goal)

                neighbor_f = tentative_g + neighbor_h

                heapq.heappush(
                    open_heap,
                    (
                        neighbor_f,
                        neighbor_h,
                        neighbor[0],
                        neighbor[1],
                    ),
                )
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


def find_nearest_reachable_goal_path(
    data,
    width,
    height,
    start,
    goals,
) -> tuple[
    tuple[int, int],
    list[tuple[int, int]],
] | None:

    if start is None or not goals:
        return None
    if not is_traversable_grid_cell(
        data=data,
        width=width,
        height=height,
        row=start[0],
        column=start[1],
    ):
        return None

    valid_goals = {
        goal
        for goal in goals
        if is_traversable_grid_cell(
            data=data,
            width=width,
            height=height,
            row=goal[0],
            column=goal[1],
        )
    }

    if not valid_goals:
        return None

    came_from = {}
    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()

        if current in valid_goals:
            path = reconstruct_grid_path(came_from, current)
            return current, path

        for neighbor in traversable_grid_neighbors(
            data=data,
            width=width,
            height=height,
            cell=current,
        ):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return None


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


def find_cluster_approach_cell_reachable(
    raw_data,
    planning_data,
    width,
    height,
    cluster,
    bfs,
    max_search_radius_cells,
) -> tuple[int, int] | None:
    """Pick a reachable standoff approach cell for a cluster."""
    # Frontier cells normally sit inside unknown-space padding, so
    # they are planning-blocked even though they are raw-free. The
    # search must therefore TRAVERSE raw-free cells (raw_data == 0),
    # but only ACCEPT a result when it is planning-free
    # (planning_data == 0) AND inside the rover's reachable
    # component (bfs['reachable']). This finds the first safe
    # reachable standoff cell outward from the frontier without
    # ever crossing raw occupied or unknown cells. The search is
    # bounded by ``max_search_radius_cells`` so a distant cluster
    # cannot map onto an unrelated cell across the map. Returns
    # None when no such cell exists within the bound.

    if not cluster or bfs is None:
        return None

    reachable = bfs['reachable']
    width = bfs['width']

    representative = (
        representative_frontier_cell(cluster)
    )

    def raw_free(cell):
        return (
            0 <= cell[0] < height
            and 0 <= cell[1] < width
            and raw_data[
                cell[0] * width + cell[1]
            ] == 0
        )

    def is_valid_approach(cell):
        return bool(
            planning_data[
                cell[0] * width + cell[1]
            ] == 0
            and reachable[
                cell[0] * width + cell[1]
            ]
        )

    # The representative itself: valid when planning-free and
    # reachable, or traversable as a raw-free seed.
    if is_valid_approach(representative):
        return representative

    visited = set()
    queue = deque()

    for cell in sorted(
        cluster,
        key=lambda member: manhattan_grid_distance(
            member, representative
        ),
    ):
        if not raw_free(cell):
            continue

        if cell not in visited:
            visited.add(cell)

            if is_valid_approach(cell):
                return cell

            queue.append((cell, 0))

    while queue:
        current, depth = queue.popleft()

        if depth >= max_search_radius_cells:
            continue

        for row_offset, column_offset in neighbor_offsets:
            neighbor = (
                current[0] + row_offset,
                current[1] + column_offset,
            )

            if neighbor in visited:
                continue

            if not raw_free(neighbor):
                continue

            visited.add(neighbor)

            if is_valid_approach(neighbor):
                return neighbor

            queue.append((neighbor, depth + 1))

    return None


def select_cluster_weighted_goal(
    bfs,
    candidate_costs,
    distance_slack_cells,
) -> tuple[
    tuple[int, int],
    list[tuple[int, int]],
] | None:
    # Pick a decisive goal among near-equally-distant candidates.
    # Uses the shared per-cycle BFS tree (bfs from
    # compute_reachable_component) for reachability, route costs,
    # and path reconstruction -- no second search is performed.
    # candidate_costs maps each candidate (row, column) to
    # (cluster_size,). The shortest reachable cost defines the cut;
    # candidates within distance_slack_cells are shortlisted and the
    # largest cluster wins. Ties break by shorter path cost, then
    # row, then column. Disconnected candidates cannot appear
    # because they are not in the BFS tree.

    if bfs is None or not candidate_costs:
        return None

    reachable = bfs['reachable']
    cost = bfs['cost']
    came_from = bfs['came_from']
    width = bfs['width']

    reachable_candidates = [
        (candidate, cluster_size)
        for candidate, cluster_size in (
            candidate_costs.items()
        )
        if reachable[
            candidate[0] * width + candidate[1]
        ]
    ]

    if not reachable_candidates:
        return None

    shortest_cost = min(
        cost[candidate]
        for candidate, _ in reachable_candidates
    )

    shortlist = [
        (candidate, cluster_size)
        for candidate, cluster_size in reachable_candidates
        if cost[candidate]
        <= shortest_cost + distance_slack_cells
    ]

    best_candidate, _ = max(
        shortlist,
        key=lambda item: (
            item[1],
            -cost[item[0]],
            -item[0][0],
            -item[0][1],
        ),
    )

    path = reconstruct_grid_path(came_from, best_candidate)

    return best_candidate, path
