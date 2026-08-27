import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rover_control.safety_logic import (
    choose_turn_direction,
    recovery_command,
    should_trigger_recovery,
    update_blocked_state,
    update_front_blocked_state,
)
from rover_control.scan_processing import nearest_valid_range_in_sector
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class ObstacleGuard(Node):
    def __init__(self):
        super().__init__('obstacle_guard')

        self.declare_parameter('stop_distance', 0.45)
        self.declare_parameter('resume_distance', 0.55)
        self.declare_parameter('sector_half_width_deg', 20.0)
        self.declare_parameter(
            'rear_sector_half_width_deg',
            30.0,
        )
        self.declare_parameter(
            'rear_stop_distance',
            0.30,
        )
        self.declare_parameter(
            'rear_resume_distance',
            0.40,
        )
        self.declare_parameter(
            'side_sector_half_width_deg',
            45.0,
        )
        self.declare_parameter('scan_timeout_s', 0.5)
        self.declare_parameter('command_timeout_s', 0.5)
        self.declare_parameter(
            'recovery.trigger_after_blocked_s',
            4.0,
        )
        self.declare_parameter('recovery.reverse_speed', 0.10)
        self.declare_parameter('recovery.reverse_duration_s', 1.5)
        self.declare_parameter('recovery.turn_speed', 0.60)
        self.declare_parameter('recovery.turn_duration_s', 2.75)

        self.stop_distance = (
            self.get_parameter('stop_distance').value
        )
        self.resume_distance = (
            self.get_parameter('resume_distance').value
        )
        self.sector_half_width = math.radians(
            self.get_parameter('sector_half_width_deg').value
        )
        self.rear_sector_half_width = math.radians(
            self.get_parameter('rear_sector_half_width_deg').value
        )
        self.rear_stop_distance = (
            self.get_parameter('rear_stop_distance').value
        )
        self.rear_resume_distance = (
            self.get_parameter('rear_resume_distance').value
        )
        self.side_sector_half_width = math.radians(
            self.get_parameter('side_sector_half_width_deg').value
        )
        self.scan_timeout_s = (
            self.get_parameter('scan_timeout_s').value
        )
        self.command_timeout_s = (
            self.get_parameter('command_timeout_s').value
        )
        self.recovery_trigger_duration_s = (
            self.get_parameter(
                'recovery.trigger_after_blocked_s'
            ).value
        )
        self.recovery_reverse_speed = (
            self.get_parameter('recovery.reverse_speed').value
        )
        self.recovery_reverse_duration_s = (
            self.get_parameter(
                'recovery.reverse_duration_s'
            ).value
        )
        self.recovery_turn_speed = (
            self.get_parameter('recovery.turn_speed').value
        )
        self.recovery_turn_duration_s = (
            self.get_parameter(
                'recovery.turn_duration_s'
            ).value
        )

        self.front_blocked = False
        self.blocked_since_time = None
        self.recovery_attempted_for_block = False

        self.rear_blocked = False
        self.rear_distance_valid = False

        self.recovery_active = False
        self.recovery_start_time = None
        self.recovery_turn_sign = 1.0

        # Set by the command watchdog; while true no recovery motion
        # may be published and control_callback must not re-enter.
        self.command_is_stale = True
        self.last_command_time = None

        self.forward_requested = False

        self.scan_is_stale = True
        self.last_scan_time = None

        self.last_scan_ranges = []
        self.last_scan_angle_min = 0.0
        self.last_scan_angle_increment = 0.0
        self.last_scan_range_min = 0.0
        self.last_scan_range_max = 0.0

        # Coordination with the frontier planner: a stuck request
        # asks for recovery; recovery_status tells the planner when
        # a maneuver ends so it can assign a fresh goal.
        self.stuck_request_pending = False
        self.requested_recovery_attempted = False

        self.command_watchdog_timer = self.create_timer(
            0.1,
            self.command_watchdog_callback,
        )

        self.control_timer = self.create_timer(
            0.1,
            self.control_callback,
        )

        self.command_subscription = self.create_subscription(
            Twist,
            '/cmd_vel_raw',
            self.command_callback,
            10,
        )

        self.command_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10,
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.scan_watchdog_timer = self.create_timer(
            0.1,
            self.scan_watchdog_callback,
        )

        self.recovery_request_subscription = (
            self.create_subscription(
                Bool,
                '/recovery_request',
                self.recovery_request_callback,
                # Reliable but volatile: a request must NOT replay if
                # the guard restarts, otherwise a long-consumed stuck
                # request would trigger an unexpected maneuver.
                10,
            )
        )

        # Recovery status is stateful: transient-local durability
        # latches the latest status so a newly started or restarted
        # Frontier node immediately learns whether recovery is
        # running. Requests stay volatile (see above) precisely
        # because they are one-shot events, not state.
        status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.recovery_status_publisher = self.create_publisher(
            Bool,
            '/recovery_status',
            status_qos,
        )

        # Announce the current (inactive) state so late joiners get
        # an authoritative baseline, then publish every transition.
        self.publish_recovery_status(False)

    def scan_callback(self, scan):
        self.last_scan_time = self.get_clock().now()
        self.scan_is_stale = False

        self.last_scan_ranges = scan.ranges
        self.last_scan_angle_min = scan.angle_min
        self.last_scan_angle_increment = scan.angle_increment
        self.last_scan_range_min = scan.range_min
        self.last_scan_range_max = scan.range_max

        nearest_distance = nearest_valid_range_in_sector(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            sector_center=0.0,
            sector_half_width=self.sector_half_width,
        )

        # The rear sector is centred on pi and can wrap across the
        # -pi/pi boundary; the sector helper normalises each beam
        # angle before comparison.
        rear_distance = (
            nearest_valid_range_in_sector(
                ranges=scan.ranges,
                angle_min=scan.angle_min,
                angle_increment=scan.angle_increment,
                range_min=scan.range_min,
                range_max=scan.range_max,
                sector_center=math.pi,
                sector_half_width=self.rear_sector_half_width,
            )
        )

        was_blocked = self.front_blocked

        self.front_blocked = update_front_blocked_state(
            nearest_distance=nearest_distance,
            was_blocked=was_blocked,
            stop_distance=self.stop_distance,
            resume_distance=self.resume_distance,
        )

        if self.front_blocked and not was_blocked:
            self.blocked_since_time = self.get_clock().now()
            self.recovery_attempted_for_block = False
            self.publish_stop()

        if not self.front_blocked:
            self.blocked_since_time = None
            self.recovery_attempted_for_block = False

        # Rear blocked state uses the same hysteresis pattern. An
        # invalid rear reading (None) marks rear data unavailable;
        # reverse recovery must not run without valid rear data.
        self.rear_blocked = update_blocked_state(
            nearest_distance=rear_distance,
            was_blocked=self.rear_blocked,
            stop_distance=self.rear_stop_distance,
            resume_distance=self.rear_resume_distance,
        )
        self.rear_distance_valid = rear_distance is not None

        if (
            self.recovery_active
            and self.recovery_phase(now=None) == 'reverse'
            and self.rear_blocked
        ):
            # A rear obstacle appeared mid-recovery: hold still
            # immediately instead of backing into it.
            self.publish_stop()

    def publish_stop(self):
        self.command_publisher.publish(Twist())

    def command_callback(self, command):
        self.last_command_time = self.get_clock().now()
        self.command_is_stale = False

        if self.recovery_active or self.command_is_stale:
            return

        self.forward_requested = command.linear.x > 0.0

        unsafe_for_forward_motion = (
            self.front_blocked or self.scan_is_stale
        )

        if unsafe_for_forward_motion and command.linear.x > 0.0:
            safe_command = Twist()

            # Twist() starts with every velocity equal to zero.
            # Preserve turning so the rover can rotate away.
            safe_command.angular.z = command.angular.z
        else:
            safe_command = command

        self.command_publisher.publish(safe_command)

    def recovery_request_callback(self, request):
        if request.data:
            # A new distinct request event arms the requested
            # recovery regardless of any previous attempt; the
            # one-shot latch is consumed when the maneuver starts.
            self.stuck_request_pending = True
            self.requested_recovery_attempted = False
        else:
            self.stuck_request_pending = False
            self.requested_recovery_attempted = False

    def control_callback(self):
        now = self.get_clock().now()

        if self.recovery_active:
            self.drive_recovery(now)
            return

        if self.command_is_stale:
            # A stale raw-command watchdog stop is authoritative:
            # never start (or restart) recovery until fresh raw
            # commands resume, and stay stopped.
            self.publish_stop()
            return

        if self.should_start_requested_recovery():
            self.start_recovery(now, requested=True)
            return

        blocked_duration_s = None

        if self.front_blocked and self.blocked_since_time is not None:
            blocked_duration = now - self.blocked_since_time
            blocked_duration_s = (
                blocked_duration.nanoseconds / 1_000_000_000
            )

        should_recover = should_trigger_recovery(
            blocked_duration_s=blocked_duration_s,
            trigger_duration_s=self.recovery_trigger_duration_s,
            forward_requested=self.forward_requested,
            recovery_already_attempted=(
                self.recovery_attempted_for_block
            ),
        )

        if not should_recover:
            return

        self.start_recovery(now, requested=False)

    def should_start_requested_recovery(self):
        """One-shot handling of a frontier stuck request."""
        if not self.stuck_request_pending:
            return False

        if self.requested_recovery_attempted:
            return False

        if self.command_is_stale:
            return False

        if self.scan_is_stale or not self.rear_distance_valid:
            # Keep the request pending; safety checks must pass
            # before any maneuver runs.
            return False

        self.requested_recovery_attempted = True
        return True

    def start_recovery(self, now, requested=False):
        self.recovery_active = True
        self.recovery_start_time = now
        self.forward_requested = False

        if requested:
            self.stuck_request_pending = False
        else:
            self.recovery_attempted_for_block = True

        self.update_turn_direction()

        self.get_logger().warning(
            'Starting escape maneuver'
        )

        self.publish_recovery_status(True)
        self.drive_recovery(now)

    def update_turn_direction(self):
        # Only valid side-sector readings may choose a direction.
        # With both sides valid the rover turns toward greater
        # clearance; with exactly one valid side that side is chosen;
        # with neither side valid there is no safe guess and recovery
        # aborts.

        left_distance = nearest_valid_range_in_sector(
            ranges=self.last_scan_ranges,
            angle_min=self.last_scan_angle_min,
            angle_increment=self.last_scan_angle_increment,
            range_min=self.last_scan_range_min,
            range_max=self.last_scan_range_max,
            sector_center=math.pi / 2.0,
            sector_half_width=self.side_sector_half_width,
        )
        right_distance = nearest_valid_range_in_sector(
            ranges=self.last_scan_ranges,
            angle_min=self.last_scan_angle_min,
            angle_increment=self.last_scan_angle_increment,
            range_min=self.last_scan_range_min,
            range_max=self.last_scan_range_max,
            sector_center=-math.pi / 2.0,
            sector_half_width=self.side_sector_half_width,
        )

        if left_distance is None and right_distance is None:
            self.recovery_turn_sign = 0.0
            return

        if left_distance is None:
            self.recovery_turn_sign = -1.0
            return

        if right_distance is None:
            self.recovery_turn_sign = 1.0
            return

        self.recovery_turn_sign = choose_turn_direction(
            left_distance=left_distance,
            right_distance=right_distance,
        )

    def recovery_phase(self, now):
        """Return 'reverse', 'turn', or None (finished)."""
        if now is None:
            now = self.get_clock().now()

        elapsed = now - self.recovery_start_time
        elapsed_s = elapsed.nanoseconds / 1_000_000_000

        if elapsed_s < self.recovery_reverse_duration_s:
            return 'reverse'

        if (
            elapsed_s
            < self.recovery_reverse_duration_s
            + self.recovery_turn_duration_s
        ):
            return 'turn'

        return None

    def drive_recovery(self, now):
        # Completion/timeout is checked first so a maneuver cannot
        # linger forever behind sensor-gated early returns.
        elapsed = now - self.recovery_start_time
        elapsed_s = elapsed.nanoseconds / 1_000_000_000

        command_pair = recovery_command(
            elapsed_s=elapsed_s,
            reverse_speed=self.recovery_reverse_speed,
            reverse_duration_s=self.recovery_reverse_duration_s,
            turn_speed=self.recovery_turn_speed,
            turn_duration_s=self.recovery_turn_duration_s,
            turn_sign=self.recovery_turn_sign,
        )

        if command_pair is None:
            self.finish_recovery(reason='finished')
            return

        if self.command_is_stale or self.scan_is_stale:
            # Stale raw commands or stale scans are authoritative:
            # zero velocity now and no further recovery motion.
            self.abort_recovery(reason='stale data')
            return

        if not self.scan_is_fresh(now):
            # Never continue any recovery motion on sensor data
            # older than the timeout; hold still until fresh scans
            # arrive.
            self.publish_stop()
            return

        phase = self.recovery_phase(now)

        if phase == 'reverse':
            # Rear validity and clearance are mandatory while
            # reversing; without them the rover holds still rather
            # than backing into the unknown.
            if not self.rear_distance_valid or self.rear_blocked:
                self.publish_stop()
                return

        if phase == 'turn':
            # A turn needs a usable direction; an aborted direction
            # choice stops the rover safely.
            if self.recovery_turn_sign == 0.0:
                self.publish_stop()
                return

        linear_x, angular_z = command_pair

        recovery_twist = Twist()
        recovery_twist.linear.x = linear_x
        recovery_twist.angular.z = angular_z

        self.command_publisher.publish(recovery_twist)

    def finish_recovery(self, reason='finished'):
        self.recovery_active = False
        self.recovery_start_time = None
        self.publish_stop()

        # Rearm the requested-recovery latch so a future distinct
        # stuck event can trigger a new maneuver. Finishing and
        # aborting both pass through here.
        self.requested_recovery_attempted = False

        self.publish_recovery_status(False)

        self.get_logger().info(
            f'Escape maneuver {reason}'
        )

    def abort_recovery(self, reason='aborted'):
        self.finish_recovery(reason=f'aborted: {reason}')

    def publish_recovery_status(self, active):
        status = Bool()
        status.data = active
        self.recovery_status_publisher.publish(status)

    def scan_is_fresh(self, now):
        """Report whether a scan arrived within the timeout."""
        if self.last_scan_time is None:
            return False

        scan_age = now - self.last_scan_time
        scan_age_s = scan_age.nanoseconds / 1_000_000_000

        return scan_age_s <= self.scan_timeout_s

    def scan_watchdog_callback(self):
        now = self.get_clock().now()

        if self.scan_is_stale and not self.scan_is_fresh(now):
            # Stay stopped while stale; do not republish.
            return

        if self.last_scan_time is None:
            return

        if not self.scan_is_fresh(now) or self.scan_is_stale:
            self.scan_is_stale = True
            self.publish_stop()

    def command_watchdog_callback(self):
        now = self.get_clock().now()

        if self.command_is_stale:
            if self.last_command_time is None:
                return

            command_age = now - self.last_command_time
            command_age_s = (
                command_age.nanoseconds / 1_000_000_000
            )

            if command_age_s > self.command_timeout_s:
                # Still stale: remain stopped and keep recovery
                # suppressed until fresh commands resume.
                if self.recovery_active:
                    self.abort_recovery(reason='command timeout')
                else:
                    self.publish_stop()
            return

        if self.last_command_time is None:
            return

        command_age = now - self.last_command_time
        command_age_s = (
            command_age.nanoseconds / 1_000_000_000
        )

        if command_age_s > self.command_timeout_s:
            self.command_is_stale = True
            self.forward_requested = False

            if self.recovery_active:
                # The raw command stream feeds recovery timing and
                # forward intent; losing it mid-maneuver makes the
                # maneuver unsafe and unrenewable. Abort so no later
                # control tick can resurrect it.
                self.abort_recovery(reason='command timeout')
            else:
                self.publish_stop()


def main(args=None):

    rclpy.init(args=args)

    node = ObstacleGuard()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
