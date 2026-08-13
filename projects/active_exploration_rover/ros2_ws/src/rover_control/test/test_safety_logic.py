from rover_control.safety_logic import update_front_blocked_state


def test_front_blocked_state_uses_hysteresis():
    stop_distance = 0.45
    resume_distance = 0.55
    blocked = False

    # Clear rover remains clear.
    blocked = update_front_blocked_state(
        nearest_distance=0.60,
        was_blocked=blocked,
        stop_distance=stop_distance,
        resume_distance=resume_distance,
    )
    assert blocked is False

    # Obstacle reaches the stop threshold.
    blocked = update_front_blocked_state(
        nearest_distance=0.45,
        was_blocked=blocked,
        stop_distance=stop_distance,
        resume_distance=resume_distance,
    )
    assert blocked is True

    # Distance increases, but not enough to resume.
    blocked = update_front_blocked_state(
        nearest_distance=0.50,
        was_blocked=blocked,
        stop_distance=stop_distance,
        resume_distance=resume_distance,
    )
    assert blocked is True

    # Obstacle reaches the resume threshold.
    blocked = update_front_blocked_state(
        nearest_distance=0.55,
        was_blocked=blocked,
        stop_distance=stop_distance,
        resume_distance=resume_distance,
    )
    assert blocked is False


def test_none_distance_is_treated_as_blocked():
    blocked = update_front_blocked_state(
        nearest_distance=None,
        was_blocked=False,
        stop_distance=0.45,
        resume_distance=0.55,
    )

    assert blocked is True
