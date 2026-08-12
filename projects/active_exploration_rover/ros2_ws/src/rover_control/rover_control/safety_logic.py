

def update_front_blocked_state(
    nearest_distance,
    was_blocked,
    stop_distance,
    resume_distance,
):
    if nearest_distance is None:
        return False

    if was_blocked:
        return nearest_distance < resume_distance

    if not was_blocked:
        return nearest_distance <= stop_distance

    return nearest_distance <= stop_distance
