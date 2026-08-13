import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.duration import Duration
from rover_control.obstacle_guard import ObstacleGuard
from sensor_msgs.msg import LaserScan


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
