import math


def normalize_beam_angle(angle):
    """Wrap an angle into [-pi, pi)."""
    return math.atan2(math.sin(angle), math.cos(angle))


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

    # Normalise both the sector bounds and each beam angle so a
    # sector centred on the rear (pi) works even though beams wrap
    # across the -pi/pi boundary.
    center = normalize_beam_angle(sector_center)

    lower = normalize_beam_angle(center - sector_half_width)
    upper = normalize_beam_angle(center + sector_half_width)

    wraps = lower > upper

    for i, distance in enumerate(ranges):
        angle = normalize_beam_angle(
            angle_min + i * angle_increment
        )

        if wraps:
            inside = angle >= lower or angle <= upper
        else:
            inside = lower <= angle <= upper

        if not inside:
            continue

        if distance == math.inf:
            distance = range_max
        elif not math.isfinite(distance):
            continue

        if not (range_min <= distance <= range_max):
            continue

        if closest is None or distance < closest:
            closest = distance

    return closest
