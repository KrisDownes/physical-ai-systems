import math


def quaternion_yaw(x, y, z, w) -> float:
    numerator = 2.0 * (w * z + x * y)
    denominator = 1.0 - 2.0 * (y ** 2 + z ** 2)

    return math.atan2(numerator, denominator)


def normalize_angle(angle) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def distance_to_goal_m(position, goal_position) -> float:
    position_x, position_y = position
    goal_x, goal_y = goal_position

    return math.hypot(goal_x - position_x, goal_y - position_y)


def bearing_to_goal_rad(position, goal_position) -> float:
    """Absolute world-frame heading pointing at the goal."""
    position_x, position_y = position
    goal_x, goal_y = goal_position

    return math.atan2(
        goal_y - position_y,
        goal_x - position_x,
    )


def heading_error_rad(yaw, position, goal_position) -> float:
    # Normalised angle difference so wraparound across the +-pi
    # boundary is handled correctly. The result lies in [0, pi].

    target_bearing = bearing_to_goal_rad(
        position, goal_position
    )

    error = normalize_angle(target_bearing - yaw)

    return abs(error)


def alignment_progress_rad(samples, goal_position) -> float:
    # Positive means the rover ended the window better aligned with
    # the goal than it started. Alternating left/right rotation
    # without net alignment yields roughly zero progress and no
    # exemption from stuck detection.

    oldest = samples[0]
    newest = samples[-1]

    initial_error = heading_error_rad(
        oldest[3], oldest[1:3], goal_position
    )
    current_error = heading_error_rad(
        newest[3], newest[1:3], goal_position
    )

    return initial_error - current_error


def prune_blacklist(
    blacklist,
    now_s,
    blacklist_duration_s,
):
    return [
        entry for entry in blacklist
        if now_s - entry[2] < blacklist_duration_s
    ]


def is_goal_blacklisted(
    goal_x,
    goal_y,
    blacklist,
    blacklist_radius_m,
) -> bool:
    for blacklisted_x, blacklisted_y, _ in blacklist:
        distance = math.hypot(
            goal_x - blacklisted_x,
            goal_y - blacklisted_y,
        )

        if distance <= blacklist_radius_m:
            return True

    return False


def is_stuck(
    progress_samples,
    goal_position,
    minimum_window_s,
    progress_threshold_m,
    alignment_threshold_rad=math.pi / 8.0,
) -> bool:
    # A sample is (time_s, x, y, yaw). The window is judged only once
    # it spans minimum_window_s seconds, so a freshly assigned goal
    # always gets a full fresh window before blacklisting. Progress is
    # measured as reduction in distance to the committed goal, not
    # first-to-last displacement. Genuine alignment counts as
    # legitimate motion: if the rover substantially reduced its
    # heading error toward the goal over the window (e.g. a ~180
    # degree turn onto the goal bearing), it gets an alignment grace
    # period. Merely oscillating left/right accumulates no alignment
    # progress and is eventually classified as stuck.

    if len(progress_samples) < 2:
        return False

    oldest_time_s = progress_samples[0][0]
    newest_time_s = progress_samples[-1][0]

    window_s = newest_time_s - oldest_time_s

    if window_s < minimum_window_s:
        return False

    oldest_position = progress_samples[0][1:3]
    newest_position = progress_samples[-1][1:3]

    initial_distance = distance_to_goal_m(
        oldest_position, goal_position
    )
    current_distance = distance_to_goal_m(
        newest_position, goal_position
    )

    progress_m = initial_distance - current_distance

    if progress_m >= progress_threshold_m:
        return False

    alignment_rad = alignment_progress_rad(
        progress_samples, goal_position
    )

    return alignment_rad < alignment_threshold_rad
