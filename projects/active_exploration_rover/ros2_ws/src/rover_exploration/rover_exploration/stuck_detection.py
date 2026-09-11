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


def route_projection(position, route):
    """Signed arc coordinate on a fixed polyline; ties prefer earlier segments."""
    best = None
    arc = 0.0
    for a, b in zip(route, route[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if not length:
            continue
        u = max(0.0, min(1.0, ((position[0]-a[0])*dx + (position[1]-a[1])*dy)/length**2))
        point = (a[0]+u*dx, a[1]+u*dy)
        item = (math.dist(position, point), arc+u*length, math.atan2(dy, dx))
        if best is None or item[0] < best[0]:
            best = item
        arc += length
    return best


def route_progress(samples, minimum_window_s, progress_threshold_m,
                   alignment_threshold_rad=math.pi/8):
    """Compare both endpoints on the OLD sample's route, never across replans.

    Samples: (sim_s, x, y, yaw, route_version, immutable_world_polyline).
    Net arc advance (not accumulated travel) rejects back-and-forth motion.
    Replanning neither clears samples nor credits a change in path length.
    """
    if not samples:
        return dict(ready=False, stuck=False, reason='no_samples')
    old, new = samples[0], samples[-1]
    route = old[5]
    a, b = route_projection(old[1:3], route), route_projection(new[1:3], route)
    progress = b[1]-a[1] if a and b else 0.0
    # One fixed local bearing prevents replan-induced alignment credit.
    alignment = (abs(normalize_angle(a[2]-old[3])) -
                 abs(normalize_angle(a[2]-new[3]))) if a else 0.0
    ready = new[0]-old[0] >= minimum_window_s
    stuck = ready and progress < progress_threshold_m and alignment < alignment_threshold_rad
    return dict(ready=ready, stuck=stuck, reason='route_no_progress' if stuck else 'route_progress_or_grace',
                window_s=new[0]-old[0], progress_m=progress, alignment_rad=alignment,
                reference_route_version=old[4], current_route_version=new[4],
                start_pose=list(old[1:4]), end_pose=list(new[1:4]),
                start_arc_m=a[1] if a else None, end_arc_m=b[1] if b else None,
                route_available=bool(a and b))
