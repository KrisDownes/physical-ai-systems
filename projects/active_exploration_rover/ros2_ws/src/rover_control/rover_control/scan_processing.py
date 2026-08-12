import math


def nearest_valid_range_in_sector(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    sector_center,
    sector_half_width,
):
    closest = None
    lower = sector_center - sector_half_width
    upper = sector_center + sector_half_width

    for i, distance in enumerate(ranges):
        angle = angle_min + i * angle_increment

        if not (lower <= angle <= upper):
            continue

        if not math.isfinite(distance):
            continue

        if not (range_min <= distance <= range_max):
            continue

        if closest is None or distance < closest:
            closest = distance

    return closest
