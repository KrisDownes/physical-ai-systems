
neighbor_offsets = (
    (-1, 0),  # north
    (1, 0),   # south
    (0, -1),  # west
    (0, 1),   # east
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
