def update_blocked_state(
    nearest_distance,
    was_blocked,
    stop_distance,
    resume_distance,
):
    if nearest_distance is None:
        return True

    if was_blocked:
        return nearest_distance < resume_distance

    if not was_blocked:
        return nearest_distance <= stop_distance

    return nearest_distance <= stop_distance


# Kept for compatibility with the existing front-sector call sites.
update_front_blocked_state = update_blocked_state


def should_trigger_recovery(
    blocked_duration_s,
    trigger_duration_s,
    forward_requested,
    recovery_already_attempted,
):
    if recovery_already_attempted:
        return False

    if not forward_requested:
        return False

    if blocked_duration_s is None:
        return False

    return blocked_duration_s >= trigger_duration_s


def choose_turn_direction(
    left_distance,
    right_distance,
) -> float:
    """Pick a positive or negative turn speed."""
    # The rover turns toward whichever side currently has more
    # clearance. When either side is unknown, turn left as before.
    if left_distance is None or right_distance is None:
        return 1.0

    if right_distance > left_distance:
        return -1.0

    return 1.0


def recovery_command(
    elapsed_s,
    reverse_speed,
    reverse_duration_s,
    turn_speed,
    turn_duration_s,
    turn_sign=1.0,
):
    if elapsed_s < 0.0:
        raise ValueError('Elapsed time cannot be negative')

    if elapsed_s < reverse_duration_s:
        return -reverse_speed, 0.0

    if elapsed_s < reverse_duration_s + turn_duration_s:
        return 0.0, turn_speed * turn_sign

    return None
