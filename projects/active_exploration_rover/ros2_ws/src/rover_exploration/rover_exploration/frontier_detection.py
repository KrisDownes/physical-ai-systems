from collections import deque
import math

neighbor_offsets = (
    (-1, 0),  # north
    (1, 0),   # south
    (0, -1),  # west
    (0, 1),   # east
)

cluster_neighbor_offsets = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def find_frontier_cells(data, width, height) -> set[tuple[int, int]]:
    coordinates: set[tuple[int, int]] = set()
    if width <= 0 or height <= 0:
        raise ValueError('Dimensions must be positive')
    if len(data) != width * height:
        raise ValueError('Invalid measurement amount')
    for row in range(height):
        for column in range(width):
            index = row * width + column
            value = data[index]
            if value != 0:
                continue
            for row_offset, column_offset in neighbor_offsets:
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if 0 <= neighbor_row < height and 0 <= neighbor_column < width:
                    neighbor_index = neighbor_row * width + neighbor_column
                    if data[neighbor_index] == -1:
                        coordinates.add((row, column))
                        break
    return coordinates


def grid_cell_center(
    row,
    column,
    resolution,
    origin_x,
    origin_y,
) -> tuple[float, float]:

    pos_x = origin_x + (column + 0.5) * resolution
    pos_y = origin_y + (row + 0.5) * resolution
    return pos_x, pos_y


def cluster_frontier_cells(
    frontier_cells,
    min_cluster_size=5,
) -> list[set[tuple[int, int]]]:

    if min_cluster_size <= 0:
        raise ValueError('Minimum cluster size must be positive')

    unvisited = set(frontier_cells)
    clusters = []

    while unvisited:
        start = unvisited.pop()
        cluster = {start}
        queue = deque([start])

        while queue:
            row, column = queue.popleft()
            for row_offset, column_offset in cluster_neighbor_offsets:
                neighbor = (row + row_offset, column + column_offset)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    cluster.add(neighbor)
                    queue.append(neighbor)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def frontier_cluster_centroid(
    cluster,
) -> tuple[float, float]:

    if not cluster:
        raise ValueError('Cannot calcualte the centroid of an empty cluster')

    row_total = 0
    column_total = 0

    for row, column in cluster:
        row_total += row
        column_total += column
    avg_row = row_total / len(cluster)
    avg_col = column_total / len(cluster)
    return avg_row, avg_col


def representative_frontier_cell(
    cluster,
) -> tuple[int, int]:

    centroid_row, centroid_column = frontier_cluster_centroid(cluster)

    representative = min(
        cluster,
        key=lambda cell: (
            (cell[0] - centroid_row) ** 2
            + (cell[1] - centroid_column) ** 2,
            cell[0],
            cell[1],
        ),
    )

    return representative


def world_point_to_grid_cell(
    world_x,
    world_y,
    resolution,
    origin_x,
    origin_y,
) -> tuple[int, int]:

    if resolution <= 0:
        raise ValueError('Resolution has to be positive')

    grid_column = (world_x - origin_x) / resolution
    grid_row = (world_y - origin_y) / resolution

    nearest_column = round(grid_column)
    nearest_row = round(grid_row)

    if math.isclose(
        grid_column,
        nearest_column,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        grid_column = nearest_column

    if math.isclose(
        grid_row,
        nearest_row,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        grid_row = nearest_row

    column = math.floor(grid_column)
    row = math.floor(grid_row)

    return row, column


def select_nearest_frontier_candidate(
    frontier_clusters,
    robot_grid_cell,
) -> tuple[int, int] | None:

    if robot_grid_cell is None or not frontier_clusters:
        return None

    robot_row, robot_column = robot_grid_cell

    candidate_cells = (
        representative_frontier_cell(cluster)
        for cluster in frontier_clusters
    )

    return min(
        candidate_cells,
        key=lambda cell: (
            (cell[0] - robot_row) ** 2
            + (cell[1] - robot_column) ** 2,
            cell[0],
            cell[1],
        ),
    )
