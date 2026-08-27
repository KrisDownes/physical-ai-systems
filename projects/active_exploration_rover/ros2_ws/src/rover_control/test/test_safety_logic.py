from rover_control.safety_logic import (
    recovery_command,
    should_trigger_recovery,
    update_front_blocked_state,
)


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


def test_recovery_triggers_after_block_duration_with_forward_request():
    assert should_trigger_recovery(
        blocked_duration_s=4.0,
        trigger_duration_s=4.0,
        forward_requested=True,
        recovery_already_attempted=False,
    ) is True


def test_recovery_does_not_trigger_before_trigger_duration():
    assert should_trigger_recovery(
        blocked_duration_s=3.9,
        trigger_duration_s=4.0,
        forward_requested=True,
        recovery_already_attempted=False,
    ) is False


def test_recovery_requires_forward_request():
    assert should_trigger_recovery(
        blocked_duration_s=10.0,
        trigger_duration_s=4.0,
        forward_requested=False,
        recovery_already_attempted=False,
    ) is False


def test_recovery_only_triggers_once_per_block():
    assert should_trigger_recovery(
        blocked_duration_s=10.0,
        trigger_duration_s=4.0,
        forward_requested=True,
        recovery_already_attempted=True,
    ) is False


def test_recovery_does_not_trigger_when_not_blocked():
    assert should_trigger_recovery(
        blocked_duration_s=None,
        trigger_duration_s=4.0,
        forward_requested=True,
        recovery_already_attempted=False,
    ) is False


def test_recovery_command_reverses_then_turns_then_finishes():
    command = recovery_command(
        elapsed_s=0.5,
        reverse_speed=0.10,
        reverse_duration_s=1.5,
        turn_speed=0.60,
        turn_duration_s=2.75,
    )

    assert command == (-0.10, 0.0)

    command = recovery_command(
        elapsed_s=2.0,
        reverse_speed=0.10,
        reverse_duration_s=1.5,
        turn_speed=0.60,
        turn_duration_s=2.75,
    )

    assert command == (0.0, 0.60)

    command = recovery_command(
        elapsed_s=4.5,
        reverse_speed=0.10,
        reverse_duration_s=1.5,
        turn_speed=0.60,
        turn_duration_s=2.75,
    )

    assert command is None


def test_recovery_command_rejects_negative_elapsed_time():
    try:
        recovery_command(
            elapsed_s=-0.1,
            reverse_speed=0.10,
            reverse_duration_s=1.5,
            turn_speed=0.60,
            turn_duration_s=2.75,
        )
        raised = False
    except ValueError:
        raised = True

    assert raised is True
