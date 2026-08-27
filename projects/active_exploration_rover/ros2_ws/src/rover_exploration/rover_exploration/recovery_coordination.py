# Pure state machine for frontier/recovery coordination.
#
# Keeps the request/active lifecycle testable without ROS, TF or map
# infrastructure. The planner must treat ``planning_blocked`` as its
# goal-assignment guard so no goal is assigned while a recovery
# request is pending or a recovery maneuver is active.


class RecoveryCoordinationState:
    """Track the frontier-side recovery request cycle."""

    def __init__(self):
        self.request_pending = False
        self.recovery_active = False

    @property
    def planning_blocked(self):
        # True while a request is pending OR recovery is active.
        return self.request_pending or self.recovery_active

    def publish_request(self):
        # Returns True only when no cycle is running; repeated
        # timer ticks for the same event are suppressed because the
        # pending flag stays set until status arrives.

        if self.planning_blocked:
            return False

        self.request_pending = True
        return True

    def on_status_active(self):
        """Transition from pending to active when status arrives."""
        was_active = self.recovery_active
        self.recovery_active = True
        self.request_pending = False
        return not was_active

    def on_status_inactive(self):
        # Returns True when this ends an active/pending cycle and
        # the progress window should be cleared.

        had_cycle = (
            self.recovery_active or self.request_pending
        )
        self.recovery_active = False
        self.request_pending = False
        return had_cycle

    def begin_new_goal(self):
        """Clear any stale cycle when a fresh goal is assigned."""
        return self.on_status_inactive()
