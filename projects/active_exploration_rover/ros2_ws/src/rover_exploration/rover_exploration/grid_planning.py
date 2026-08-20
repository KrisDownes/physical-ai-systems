import heapq

neighbor_offsets = (
    (-1, 0),  # north
    (1, 0),   # south
    (0, -1),  # west
    (0, 1),   # east
    )


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
