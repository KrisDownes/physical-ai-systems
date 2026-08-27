from rover_exploration.recovery_coordination import (
    RecoveryCoordinationState,
)


def test_idle_state_allows_planning():
    state = RecoveryCoordinationState()

    assert state.planning_blocked is False


def test_request_publication_blocks_planning_immediately():
    # The pending flag is set before any status arrives, closing the
    # race where a goal could be assigned between the request and
    # the guard's active status.
    state = RecoveryCoordinationState()

    published = state.publish_request()

    assert published is True
    assert state.request_pending is True
    assert state.planning_blocked is True


def test_active_status_blocks_planning():
    state = RecoveryCoordinationState()
    state.publish_request()

    changed = state.on_status_active()

    assert changed is True
    assert state.recovery_active is True
    assert state.request_pending is False
    assert state.planning_blocked is True


def test_inactive_status_reenables_planning_and_rearms():
    state = RecoveryCoordinationState()
    state.publish_request()
    state.on_status_active()

    ended = state.on_status_inactive()

    assert ended is True
    assert state.planning_blocked is False
    assert state.request_pending is False
    assert state.recovery_active is False

    # Rearmed: a future distinct event can publish again.
    assert state.publish_request() is True


def test_abort_also_reenables_planning():
    state = RecoveryCoordinationState()
    state.publish_request()
    state.on_status_active()

    ended = state.on_status_inactive()

    assert ended is True
    assert state.planning_blocked is False


def test_single_stuck_event_does_not_repeat_requests():
    state = RecoveryCoordinationState()

    assert state.publish_request() is True

    # Repeated stuck-timer ticks for the same event are suppressed.
    for _ in range(5):
        assert state.publish_request() is False

    assert state.request_pending is True


def test_full_lifecycle_two_consecutive_cycles():
    # idle -> pending -> active -> finished -> idle/rearmed ->
    # second pending -> second active -> aborted -> idle/rearmed.
    state = RecoveryCoordinationState()

    # idle
    assert state.planning_blocked is False

    # first request pending (planning blocks immediately)
    assert state.publish_request() is True
    assert state.planning_blocked is True

    # recovery active
    assert state.on_status_active() is True
    assert state.planning_blocked is True

    # recovery finished -> idle/rearmed; progress window resets
    assert state.on_status_inactive() is True
    assert state.planning_blocked is False

    # At the node level no further request is emitted for this
    # event because the goal was cleared on blacklisting, so the
    # stuck timer never fires again until a NEW goal exists.
    # Simulate that here: with no new stuck event the state simply
    # stays idle; a fresh event would call publish_request() again.

    # second request pending (new distinct event)
    assert state.publish_request() is True
    assert state.planning_blocked is True

    # second recovery active
    assert state.on_status_active() is True
    assert state.planning_blocked is True

    # recovery aborted -> idle/rearmed again
    assert state.on_status_inactive() is True
    assert state.planning_blocked is False
