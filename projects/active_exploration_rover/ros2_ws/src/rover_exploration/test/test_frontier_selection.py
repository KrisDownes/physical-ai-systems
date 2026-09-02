"""Tests for frontier detection, clustering, and coordinates."""

from rover_exploration.frontier_selection import (
    cluster_frontier_cells,
    find_frontier_cells,
    grid_cell_center,
    representative_frontier_cell,
    world_point_to_grid_cell,
)


def test_frontier_is_free_cell_adjacent_to_unknown():
    data = [
        100, 100, 100,
        100, 0, -1,
        100, 100, 100,
    ]
    assert find_frontier_cells(data, 3, 3) == {(1, 1)}


def test_clustering_uses_eight_connectivity_and_minimum_size():
    cells = {(0, 0), (1, 1), (2, 2), (8, 8)}
    assert cluster_frontier_cells(cells, min_cluster_size=3) == [
        {(0, 0), (1, 1), (2, 2)}
    ]


def test_representative_cell_is_deterministic():
    cluster = {(0, 0), (0, 2), (2, 0), (2, 2)}
    assert representative_frontier_cell(cluster) == (0, 0)


def test_world_grid_conversion_uses_containing_cell():
    assert grid_cell_center(2, 3, 0.5, -1.0, 4.0) == (0.75, 5.25)
    assert world_point_to_grid_cell(0.75, 5.25, 0.5, -1.0, 4.0) == (2, 3)
    assert world_point_to_grid_cell(-1.01, 3.99, 0.5, -1.0, 4.0) == (-1, -1)
