"""Pure frontier detection, clustering, and coordinate selection helpers."""

from collections import deque
from dataclasses import dataclass
import math

from rover_exploration.grid_planning import manhattan_grid_distance
from rover_exploration.grid_planning import reconstruct_grid_path


CARDINAL_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CLUSTER_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def find_frontier_cells(data, width, height):
    """Return free cells that share a cardinal edge with unknown space."""
    if width <= 0 or height <= 0:
        raise ValueError('Dimensions must be positive')
    if len(data) != width * height:
        raise ValueError('Invalid measurement amount')

    frontiers = set()
    for row in range(height):
        for column in range(width):
            if data[row * width + column] != 0:
                continue
            for row_offset, column_offset in CARDINAL_NEIGHBORS:
                neighbor_row = row + row_offset
                neighbor_column = column + column_offset
                if (
                    0 <= neighbor_row < height
                    and 0 <= neighbor_column < width
                    and data[neighbor_row * width + neighbor_column] == -1
                ):
                    frontiers.add((row, column))
                    break
    return frontiers


def cluster_frontier_cells(frontier_cells, min_cluster_size=5):
    """Return deterministic eight-connected frontier components."""
    if min_cluster_size <= 0:
        raise ValueError('Minimum cluster size must be positive')

    unvisited = set(frontier_cells)
    clusters = []
    while unvisited:
        start = min(unvisited)
        unvisited.remove(start)
        cluster = {start}
        queue = deque([start])
        while queue:
            row, column = queue.popleft()
            for row_offset, column_offset in CLUSTER_NEIGHBORS:
                neighbor = (row + row_offset, column + column_offset)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    cluster.add(neighbor)
                    queue.append(neighbor)
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    return clusters


def grid_cell_center(row, column, resolution, origin_x, origin_y):
    """Return the world-frame center of a grid cell."""
    return (
        origin_x + (column + 0.5) * resolution,
        origin_y + (row + 0.5) * resolution,
    )


def representative_frontier_cell(cluster):
    """Return the cluster member nearest its centroid, deterministically."""
    if not cluster:
        raise ValueError('Cannot choose from an empty cluster')
    center_row = sum(row for row, _ in cluster) / len(cluster)
    center_column = sum(column for _, column in cluster) / len(cluster)
    return min(
        cluster,
        key=lambda cell: (
            (cell[0] - center_row) ** 2 + (cell[1] - center_column) ** 2,
            cell[0],
            cell[1],
        ),
    )


def world_point_to_grid_cell(
    world_x, world_y, resolution, origin_x, origin_y
):
    """Convert a world point to the containing occupancy-grid cell."""
    if resolution <= 0:
        raise ValueError('Resolution has to be positive')
    grid_column = (world_x - origin_x) / resolution
    grid_row = (world_y - origin_y) / resolution
    nearest_column = round(grid_column)
    nearest_row = round(grid_row)
    if math.isclose(grid_column, nearest_column, rel_tol=0.0, abs_tol=1e-9):
        grid_column = nearest_column
    if math.isclose(grid_row, nearest_row, rel_tol=0.0, abs_tol=1e-9):
        grid_row = nearest_row
    return math.floor(grid_row), math.floor(grid_column)


@dataclass(frozen=True)
class CandidateSet:
    """Reachable approaches and facts produced while finding them."""

    sizes: dict
    anchors: dict
    duplicates: int
    unreachable_clusters: int


@dataclass(frozen=True)
class CandidateFilter:
    """Eligible approaches and exact rejection counts."""

    eligible: list
    visited: int
    temporary: int
    permanent: int
    retry_exhausted: int
    too_close: int


def find_reachable_approach(
    raw_data,
    planning_data,
    width,
    height,
    cluster,
    bfs,
    max_search_radius_cells,
    excluded_cells=None,
):
    """Return the first bounded safe approach in the rover component."""
    if not cluster or bfs is None:
        return None

    excluded_cells = excluded_cells or set()
    representative = representative_frontier_cell(cluster)
    reachable = bfs['reachable']

    def raw_free(cell):
        return (
            0 <= cell[0] < height
            and 0 <= cell[1] < width
            and raw_data[cell[0] * width + cell[1]] == 0
        )

    def valid(cell):
        index = cell[0] * width + cell[1]
        return (
            planning_data[index] == 0
            and reachable[index]
            and cell not in excluded_cells
        )

    if valid(representative):
        return representative

    visited = set()
    queue = deque()
    for cell in sorted(
        cluster,
        key=lambda item: manhattan_grid_distance(item, representative),
    ):
        if not raw_free(cell) or cell in visited:
            continue
        visited.add(cell)
        if valid(cell):
            return cell
        queue.append((cell, 0))

    while queue:
        current, depth = queue.popleft()
        if depth >= max_search_radius_cells:
            continue
        for row_offset, column_offset in CARDINAL_NEIGHBORS:
            neighbor = current[0] + row_offset, current[1] + column_offset
            if neighbor in visited or not raw_free(neighbor):
                continue
            visited.add(neighbor)
            if valid(neighbor):
                return neighbor
            queue.append((neighbor, depth + 1))
    return None


def build_reachable_candidates(
    raw_data,
    planning_data,
    width,
    height,
    clusters,
    bfs,
    max_search_radius_cells,
    active_anchor=None,
    attempted_cells=None,
):
    """Build one deduplicated reachable approach per frontier cluster."""
    sizes = {}
    anchors = {}
    duplicates = 0
    unreachable = 0
    attempted_cells = attempted_cells or set()
    for cluster in clusters:
        anchor = active_anchor if active_anchor in cluster else min(cluster)
        approach = find_reachable_approach(
            raw_data,
            planning_data,
            width,
            height,
            cluster,
            bfs,
            max_search_radius_cells,
            attempted_cells if anchor == active_anchor else None,
        )
        if approach is None:
            unreachable += 1
            continue
        if approach in sizes:
            duplicates += 1
        if len(cluster) > sizes.get(approach, 0):
            sizes[approach] = len(cluster)
            anchors[approach] = anchor
    return CandidateSet(sizes, anchors, duplicates, unreachable)


def filter_candidates(
    candidates,
    memory,
    resolution,
    origin_x,
    origin_y,
    robot_world,
    now_s,
    temporary_radius_m,
    visited_radius_m,
    goal_reached_distance_m,
    active_anchor=None,
    fresh_approaches=0,
    maximum_fresh_approaches=0,
):
    """Apply memory, retry, and distance exclusions to reachable approaches."""
    eligible = []
    visited = temporary = permanent = exhausted = too_close = 0
    for cell in candidates.sizes:
        x, y = grid_cell_center(
            cell[0], cell[1], resolution, origin_x, origin_y
        )
        reason = memory.exclusion_reason(
            x, y, now_s, temporary_radius_m, visited_radius_m
        )
        if reason == 'temporary':
            temporary += 1
        elif reason == 'permanent':
            permanent += 1
        elif reason == 'visited':
            visited += 1
        elif (
            candidates.anchors[cell] == active_anchor
            and fresh_approaches >= maximum_fresh_approaches
        ):
            exhausted += 1
        elif math.hypot(x - robot_world[0], y - robot_world[1]) <= (
            goal_reached_distance_m
        ):
            too_close += 1
        else:
            eligible.append(cell)
    return CandidateFilter(
        eligible, visited, temporary, permanent, exhausted, too_close
    )


def select_weighted_goal(bfs, candidates, eligible, distance_slack_cells):
    """Prefer larger clusters among approaches with near-shortest paths."""
    if bfs is None or not eligible:
        return None
    costs = bfs['cost']
    reachable = [cell for cell in eligible if cell in costs]
    if not reachable:
        return None
    shortest = min(costs[cell] for cell in reachable)
    shortlist = [
        cell for cell in reachable
        if costs[cell] <= shortest + distance_slack_cells
    ]
    selected = max(
        shortlist,
        key=lambda cell: (
            candidates.sizes[cell], -costs[cell], -cell[0], -cell[1]
        ),
    )
    return selected, reconstruct_grid_path(bfs['came_from'], selected)
