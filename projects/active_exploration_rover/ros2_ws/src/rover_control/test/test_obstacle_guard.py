import math

from geometry_msgs.msg import Twist
import pytest
import rclpy
from rclpy.duration import Duration
from rover_control.obstacle_guard import ObstacleGuard
from sensor_msgs.msg import LaserScan


def make_blocked_scan():
    # A full-circle style scan: obstacle dead ahead, rear and sides
    # clear so recovery is allowed to run.
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = math.pi / 8.0
    scan.range_min = 0.1
    scan.range_max = 10.0
    # Beams from -pi to pi; only the forward beam is blocked.
    scan.ranges = (
        [10.0] * 8
        + [0.30]
        + [10.0] * 7
    )
    return scan


def make_rear_blocked_scan():
    # Same as a blocked-front scan but with an obstacle directly
    # behind the rover.
    scan = make_blocked_scan()
    scan.ranges[0] = 0.20  # angle -pi == rear
    scan.ranges[-1] = 0.20  # angle just under +pi == rear
    return scan


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        # Store each published message.
        self.messages.append(message)


def test_startup_blocks_forward_motion_before_first_scan():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        requested_command = Twist()
        requested_command.linear.x = 0.25
        requested_command.angular.z = 0.4

        node.command_callback(requested_command)

        assert len(recorder.messages) == 1
        published_command = recorder.messages[0]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.4

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_fresh_scan_allows_forward_motion():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.front_blocked = False

        requested_command = Twist()
        requested_command.linear.x = 0.25
        requested_command.angular.z = 0.4

        node.command_callback(requested_command)

        assert len(recorder.messages) == 1

        published_command = recorder.messages[0]

        assert published_command.linear.x == 0.25
        assert published_command.angular.z == 0.4

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_near_obstacle_immediately_publishes_stop():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        scan = LaserScan()
        scan.angle_min = -0.35
        scan.angle_increment = 0.35
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [2.0, 0.30, 2.0]

        node.scan_callback(scan)

        assert node.scan_is_stale is False
        assert node.front_blocked is True
        assert len(recorder.messages) == 1

        published_command = recorder.messages[0]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_persistent_obstacle_does_not_republish_stop():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        scan = LaserScan()
        scan.angle_min = -0.35
        scan.angle_increment = 0.35
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [2.0, 0.30, 2.0]

        node.scan_callback(scan)
        node.scan_callback(scan)

        assert node.front_blocked is True
        assert len(recorder.messages) == 1

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_watchdog_publishes_stop_when_scan_becomes_stale():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.last_scan_time = (
            node.get_clock().now()
            - Duration(seconds=node.scan_timeout_s + 0.1)
        )
        node.scan_watchdog_callback()

        assert node.scan_is_stale is True
        assert len(recorder.messages) == 1

        published_command = recorder.messages[0]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_watchdog_does_not_republish_stop_while_already_stale():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.last_scan_time = (
            node.get_clock().now()
            - Duration(seconds=node.scan_timeout_s + 0.1)
        )

        node.scan_watchdog_callback()
        node.scan_watchdog_callback()

        assert node.scan_is_stale is True
        assert len(recorder.messages) == 1

        published_command = recorder.messages[0]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_invalid_scan_immediately_publishes_stop():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        scan = LaserScan()
        scan.angle_min = -0.35
        scan.angle_increment = 0.35
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [math.nan, math.nan, math.nan]

        node.scan_callback(scan)

        assert node.scan_is_stale is False
        assert node.front_blocked is True
        assert len(recorder.messages) == 1

        published_command = recorder.messages[0]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_clear_scan_recovers_after_invalid_scan():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        scan = LaserScan()
        scan.angle_min = -0.35
        scan.angle_increment = 0.35
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [math.nan, math.nan, math.nan]

        node.scan_callback(scan)

        assert node.front_blocked is True
        assert len(recorder.messages) == 1

        scan.ranges = [math.inf, math.inf, math.inf]

        node.scan_callback(scan)

        assert node.scan_is_stale is False
        assert node.front_blocked is False

        # Clearing the blocked state does not itself publish a motion command.
        assert len(recorder.messages) == 1

        requested_command = Twist()
        requested_command.linear.x = 0.25
        requested_command.angular.z = 0.4

        node.command_callback(requested_command)

        assert len(recorder.messages) == 2

        published_command = recorder.messages[1]

        assert published_command.linear.x == 0.25
        assert published_command.angular.z == 0.4

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_starts_after_persistent_block_and_reverses():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())
        assert node.front_blocked is True

        # The follower keeps requesting forward motion while blocked.
        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        # Simulate the block persisting past the trigger duration.
        node.blocked_since_time = (
            node.get_clock().now()
            - Duration(
                seconds=node.recovery_trigger_duration_s + 0.1
            )
        )

        recorder.messages.clear()
        node.control_callback()

        assert node.recovery_active is True

        # First recovery command reverses straight back.
        published_command = recorder.messages[-1]

        assert published_command.linear.x == pytest.approx(
            -node.recovery_reverse_speed
        )
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_turns_after_reverse_phase():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_callback(make_blocked_scan())

        # A fresh raw command marks the command stream alive.
        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.blocked_since_time = (
            node.get_clock().now()
            - Duration(seconds=100.0)
        )
        node.forward_requested = True
        node.control_callback()

        # Jump to the middle of the turn phase.
        node.recovery_start_time = (
            node.get_clock().now()
            - Duration(
                seconds=(
                    node.recovery_reverse_duration_s
                    + node.recovery_turn_duration_s / 2.0
                )
            )
        )

        recorder.messages.clear()
        node.control_callback()

        published_command = recorder.messages[-1]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == pytest.approx(
            node.recovery_turn_speed
        )

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_finishes_and_does_not_restart_for_same_block():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_callback(make_blocked_scan())

        # A fresh raw command marks the command stream alive.
        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.blocked_since_time = (
            node.get_clock().now()
            - Duration(seconds=100.0)
        )
        node.forward_requested = True
        node.control_callback()
        assert node.recovery_active is True

        # Jump past the end of the maneuver.
        node.recovery_start_time = (
            node.get_clock().now()
            - Duration(
                seconds=(
                    node.recovery_reverse_duration_s
                    + node.recovery_turn_duration_s
                    + 1.0
                )
            )
        )

        node.control_callback()

        assert node.recovery_active is False

        last_message = recorder.messages[-1]

        assert last_message.linear.x == 0.0
        assert last_message.angular.z == 0.0

        # Still blocked and still requesting forward motion,
        # but no second recovery for the same block.
        node.forward_requested = True
        marker_index = len(recorder.messages)
        node.control_callback()

        assert node.recovery_active is False
        assert len(recorder.messages) == marker_index

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_swallows_raw_commands_while_active():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_callback(make_blocked_scan())

        # A fresh raw command marks the command stream alive.
        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.start_recovery(node.get_clock().now())

        recorder.messages.clear()

        requested_command = Twist()
        requested_command.linear.x = 0.25
        requested_command.angular.z = 0.4

        node.command_callback(requested_command)

        # No pass-through of follower commands during recovery.
        assert len(recorder.messages) == 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_clearing_block_resets_recovery_attempt():
    rclpy.init()
    node = ObstacleGuard()

    try:
        node.scan_callback(make_blocked_scan())
        assert node.front_blocked is True

        node.start_recovery(node.get_clock().now())
        assert node.recovery_attempted_for_block is True

        clear_scan = make_blocked_scan()
        clear_scan.ranges = [10.0] * 16
        node.scan_callback(clear_scan)

        assert node.front_blocked is False
        assert node.blocked_since_time is None
        assert node.recovery_attempted_for_block is False

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_rear_obstacle_prevents_reverse_recovery():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_rear_blocked_scan())

        assert node.front_blocked is True
        assert node.rear_blocked is True

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.blocked_since_time = (
            node.get_clock().now()
            - Duration(seconds=100.0)
        )
        node.forward_requested = True
        node.control_callback()

        assert node.recovery_active is True

        published_command = recorder.messages[-1]

        # No negative linear velocity while the rear is blocked.
        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_invalid_rear_scan_prevents_reverse_recovery():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        # Scan covers only the front; rear data is unavailable.
        scan = LaserScan()
        scan.angle_min = -0.35
        scan.angle_increment = 0.35
        scan.range_min = 0.1
        scan.range_max = 10.0
        scan.ranges = [2.0, 0.30, 2.0]

        node.scan_is_stale = False
        node.scan_callback(scan)

        assert node.rear_distance_valid is False

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.blocked_since_time = (
            node.get_clock().now()
            - Duration(seconds=100.0)
        )
        node.forward_requested = True
        node.control_callback()

        assert node.recovery_active is True

        published_command = recorder.messages[-1]

        # Reversing blind is unsafe: hold still instead.
        assert published_command.linear.x == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_stale_scan_halts_already_active_recovery():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.start_recovery(node.get_clock().now())
        assert node.recovery_active is True

        # Simulate the LiDAR going silent past the timeout.
        node.last_scan_time = (
            node.get_clock().now()
            - Duration(seconds=node.scan_timeout_s + 0.5)
        )

        recorder.messages.clear()
        node.drive_recovery(node.get_clock().now())

        published_command = recorder.messages[-1]

        assert published_command.linear.x == 0.0
        assert published_command.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_turn_direction_follows_side_clearance():
    from rover_control.safety_logic import (
        choose_turn_direction as direction,
    )

    # More clearance on the right: turn right (negative).
    assert direction(1.0, 3.0) == -1.0

    # More clearance on the left: turn left (positive).
    assert direction(3.0, 1.0) == 1.0

    # Unknown sides default to turning left.
    assert direction(None, 2.0) == 1.0
    assert direction(None, None) == 1.0


def make_full_clear_scan():
    # 16-beam full-circle scan, everything far away.
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = math.pi / 8.0
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = [10.0] * 16
    return scan


def test_turn_direction_requires_valid_sides():
    rclpy.init()
    node = ObstacleGuard()

    try:
        node.last_scan_angle_min = -math.pi
        node.last_scan_angle_increment = math.pi / 8.0
        node.last_scan_range_min = 0.1
        node.last_scan_range_max = 10.0

        def sector_indices(sector_center):
            lo = math.atan2(
                math.sin(
                    sector_center - math.pi / 4.0
                ),
                math.cos(
                    sector_center - math.pi / 4.0
                ),
            )
            hi = math.atan2(
                math.sin(
                    sector_center + math.pi / 4.0
                ),
                math.cos(
                    sector_center + math.pi / 4.0
                ),
            )
            indices = []

            for index in range(16):
                angle = -math.pi + (
                    index * math.pi / 8.0
                )
                wrapped = math.atan2(
                    math.sin(angle), math.cos(angle)
                )

                if lo <= wrapped <= hi:
                    indices.append(index)

            return indices

        left_indices = sector_indices(math.pi / 2.0)
        right_indices = sector_indices(-math.pi / 2.0)

        # Only LEFT sector valid: turn left.
        ranges = [math.nan] * 16

        for index in left_indices:
            ranges[index] = 10.0

        node.last_scan_ranges = ranges
        node.update_turn_direction()
        assert node.recovery_turn_sign == 1.0

        # Only RIGHT sector valid: turn right.
        ranges = [math.nan] * 16

        for index in right_indices:
            ranges[index] = 10.0

        node.last_scan_ranges = ranges
        node.update_turn_direction()
        assert node.recovery_turn_sign == -1.0

        # Both valid, left clearer: turn left.
        ranges = [10.0] * 16

        for index in right_indices:
            ranges[index] = 2.0

        node.last_scan_ranges = ranges
        node.update_turn_direction()
        assert node.recovery_turn_sign == 1.0

        # Both valid, right clearer: turn right.
        ranges = [10.0] * 16

        for index in left_indices:
            ranges[index] = 2.0

        node.last_scan_ranges = ranges
        node.update_turn_direction()
        assert node.recovery_turn_sign == -1.0

        # Neither valid: no safe guess.
        node.last_scan_ranges = [math.nan] * 16
        node.update_turn_direction()
        assert node.recovery_turn_sign == 0.0

        # Zero sign produces no turning command during recovery.
        from rover_control.safety_logic import (
            recovery_command,
        )

        command = recovery_command(
            elapsed_s=2.0,
            reverse_speed=node.recovery_reverse_speed,
            reverse_duration_s=(
                node.recovery_reverse_duration_s
            ),
            turn_speed=node.recovery_turn_speed,
            turn_duration_s=(
                node.recovery_turn_duration_s
            ),
            turn_sign=0.0,
        )

        assert command == (0.0, 0.0)

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_status_qos_is_transient_local():
    from rclpy.qos import QoSDurabilityPolicy

    rclpy.init()
    node = ObstacleGuard()

    try:
        qos = node.recovery_status_publisher.qos_profile

        assert qos.durability == (
            QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        assert qos.reliability.value == 1  # RELIABLE

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_command_watchdog_abort_is_authoritative_over_recovery():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.blocked_since_time = (
            node.get_clock().now()
            - Duration(seconds=100.0)
        )
        node.forward_requested = True
        node.control_callback()
        assert node.recovery_active is True

        # The raw-command stream goes silent past the timeout and
        # the watchdog runs.
        node.last_command_time = (
            node.get_clock().now()
            - Duration(seconds=node.command_timeout_s + 0.5)
        )

        recorder.messages.clear()
        node.command_watchdog_callback()

        assert node.recovery_active is False

        # A later control tick must NOT resurrect the maneuver or
        # publish any nonzero command.
        marker_index = len(recorder.messages)
        node.control_callback()

        assert node.recovery_active is False
        assert len(recorder.messages) == marker_index + 1

        stop_message = recorder.messages[-1]

        assert stop_message.linear.x == 0.0
        assert stop_message.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_timeout_with_invalid_rear_data_finishes():
    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        node.start_recovery(node.get_clock().now())
        assert node.recovery_active is True

        # Rear data became invalid and stays invalid, but the
        # maneuver's total duration has fully elapsed: the rover
        # must not stay stuck in recovery forever.
        node.rear_distance_valid = False
        node.recovery_start_time = (
            node.get_clock().now()
            - Duration(
                seconds=(
                    node.recovery_reverse_duration_s
                    + node.recovery_turn_duration_s
                    + 1.0
                )
            )
        )

        recorder.messages.clear()
        node.control_callback()

        assert node.recovery_active is False

        last_message = recorder.messages[-1]

        assert last_message.linear.x == 0.0
        assert last_message.angular.z == 0.0

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_recovery_request_triggers_and_completes_one_shot():
    from std_msgs.msg import Bool

    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        request = Bool()
        request.data = True

        node.recovery_request_callback(request)

        node.control_callback()
        assert node.recovery_active is True
        assert node.stuck_request_pending is False

        # Jump past the end of the maneuver.
        node.recovery_start_time = (
            node.get_clock().now()
            - Duration(
                seconds=(
                    node.recovery_reverse_duration_s
                    + node.recovery_turn_duration_s
                    + 1.0
                )
            )
        )
        node.control_callback()

        assert node.recovery_active is False

        # Without a NEW request event, no retrigger may occur.
        marker_index = len(recorder.messages)
        node.control_callback()

        assert node.recovery_active is False
        assert len(recorder.messages) == marker_index

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_second_distinct_request_starts_second_recovery():
    """Real two-recovery sequence with no manual flag mutation."""
    from std_msgs.msg import Bool

    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        first_request = Bool()
        first_request.data = True

        # 1. First True request.
        node.recovery_request_callback(first_request)

        # 2. Recovery starts.
        node.control_callback()
        assert node.recovery_active is True
        assert node.stuck_request_pending is False

        # 3. Advance past completion; recovery finishes.
        node.recovery_start_time = (
            node.get_clock().now()
            - Duration(
                seconds=(
                    node.recovery_reverse_duration_s
                    + node.recovery_turn_duration_s
                    + 1.0
                )
            )
        )
        recorder.messages.clear()
        node.control_callback()

        assert node.recovery_active is False

        # Finishing must have rearmed the latch.
        assert node.requested_recovery_attempted is False

        marker_index = len(recorder.messages)
        node.control_callback()

        # The consumed first request cannot retrigger on its own.
        assert len(recorder.messages) == marker_index

        # 4. A second distinct True request arrives.
        second_request = Bool()
        second_request.data = True

        node.recovery_request_callback(second_request)

        # 5. Recovery starts again.
        node.control_callback()
        assert node.recovery_active is True

        # 6. Neither request retriggers more than once without a
        # new request event: finish and tick again.
        node.recovery_start_time = (
            node.get_clock().now()
            - Duration(
                seconds=(
                    node.recovery_reverse_duration_s
                    + node.recovery_turn_duration_s
                    + 1.0
                )
            )
        )
        node.control_callback()
        assert node.recovery_active is False

        marker_index = len(recorder.messages)
        node.control_callback()

        assert node.recovery_active is False
        assert len(recorder.messages) == marker_index

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_aborted_requested_recovery_rearms_latch():
    from std_msgs.msg import Bool

    rclpy.init()
    node = ObstacleGuard()
    recorder = RecordingPublisher()

    node.command_publisher = recorder

    try:
        node.scan_is_stale = False
        node.scan_callback(make_blocked_scan())

        requested_command = Twist()
        requested_command.linear.x = 0.25
        node.command_callback(requested_command)

        request = Bool()
        request.data = True

        node.recovery_request_callback(request)
        node.control_callback()
        assert node.recovery_active is True
        assert node.requested_recovery_attempted is True

        # Force an abort via stale scan data.
        node.scan_is_stale = True
        recorder.messages.clear()
        node.drive_recovery(node.get_clock().now())

        assert node.recovery_active is False

        # Aborting must also rearm the latch.
        assert node.requested_recovery_attempted is False

        # A subsequent distinct request starts again; the scan
        # must be fresh for safety checks to pass.
        node.scan_is_stale = False
        node.last_scan_time = node.get_clock().now()

        node.recovery_request_callback(request)
        node.control_callback()

        assert node.recovery_active is True

    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_rear_sector_wraps_across_pi_boundary():
    from rover_control.scan_processing import (
        nearest_valid_range_in_sector,
    )

    # A beam at exactly +pi/2-step just below +pi and one just above
    # -pi both belong to the rear sector centred on pi.
    ranges = [10.0] * 16
    ranges[15] = 0.4   # angle_min=-pi, last beam ~ +7pi/8... near rear
    ranges[0] = 0.25   # angle -pi (rear)

    result = nearest_valid_range_in_sector(
        ranges=ranges,
        angle_min=-math.pi,
        angle_increment=math.pi / 8.0,
        range_min=0.1,
        range_max=10.0,
        sector_center=math.pi,
        sector_half_width=math.pi / 4.0,
    )

    # The wrapped rear sector must see the close beam at -pi.
    assert result == 0.25
