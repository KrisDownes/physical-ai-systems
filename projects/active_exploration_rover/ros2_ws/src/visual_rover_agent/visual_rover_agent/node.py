"""Thin ROS adapter for the bounded action state machine."""

import math

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

from sensor_msgs.msg import LaserScan

from std_msgs.msg import String

from visual_rover_agent.parser import (
    CommandError,
    parse_command,
    serialize_status,
)
from visual_rover_agent.state_machine import ActionMachine, Limits


def sector_minimum(scan, center, half_width):
    """Find the nearest valid return in an angular sector."""
    values = []
    for index, distance in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        offset = math.atan2(
            math.sin(angle - center), math.cos(angle - center))
        if abs(offset) <= half_width and not math.isnan(distance):
            if math.isinf(distance) and distance > 0:
                values.append(distance)
            elif scan.range_min <= distance <= scan.range_max:
                values.append(distance)
    return min(values) if values else None


class AgentExecutor(Node):
    """Connect agent commands and feedback to the pure action machine."""

    def __init__(self):
        """Validate parameters and connect ROS interfaces."""
        super().__init__('agent_executor')
        defaults = {
            'maximum_drive_distance_m': 0.50,
            'maximum_turn_angle_deg': 90.0,
            'maximum_linear_speed_mps': 0.15,
            'maximum_angular_speed_radps': 0.60,
            'obstacle_stop_distance_m': 0.25,
            'command_timeout_s': 10.0,
            'odometry_staleness_timeout_s': 0.5,
            'scan_staleness_timeout_s': 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {
            name: float(self.get_parameter(name).value)
            for name in defaults
        }
        invalid = any(
            not math.isfinite(value) or value <= 0.0
            for value in values.values()
        )
        if invalid:
            raise ValueError(
                'all agent limits and timeouts must be finite and positive')
        self.maximum_drive = values.pop('maximum_drive_distance_m')
        self.maximum_turn = values.pop('maximum_turn_angle_deg')
        self.machine = ActionMachine(Limits(**values))
        self.velocity_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self.status_publisher = self.create_publisher(
            String, '/agent_status', 10)
        self.create_subscription(
            String, '/agent_command', self.command_callback, 10)
        self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback,
            qos_profile_sensor_data)
        self.create_timer(0.05, self.control_callback)

    def now_s(self):
        """Return current ROS time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def command_callback(self, message):
        """Parse and submit one strict JSON command."""
        try:
            command = parse_command(
                message.data, self.maximum_drive, self.maximum_turn)
        except CommandError as error:
            self.publish_status(error.command_id, 'rejected', error.reason)
            self.publish_velocity((0.0, 0.0))
            return
        events, velocity = self.machine.submit(command, self.now_s())
        self.publish(events, velocity)

    def odom_callback(self, message):
        """Forward planar odometry to the machine."""
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.machine.update_odometry(p.x, p.y, yaw, self.now_s())

    def scan_callback(self, message):
        """Reduce the 360-degree scan to safety clearances."""
        front = sector_minimum(message, 0.0, math.radians(25.0))
        rear = sector_minimum(message, math.pi, math.radians(30.0))
        all_around = sector_minimum(message, 0.0, math.pi)
        self.machine.update_scan(front, rear, all_around, self.now_s())

    def control_callback(self):
        """Advance control without sleeps using ROS time."""
        self.publish(*self.machine.tick(self.now_s()))

    def publish(self, events, velocity):
        """Publish velocity before any resulting status transitions."""
        self.publish_velocity(velocity)
        for event in events:
            self.publish_status(event.command_id, event.state, event.reason)

    def publish_velocity(self, velocity):
        """Publish the requested planar velocity."""
        message = Twist()
        message.linear.x, message.angular.z = velocity
        self.velocity_publisher.publish(message)

    def publish_status(self, command_id, state, reason):
        """Publish one strict JSON status."""
        message = String()
        message.data = serialize_status(
            command_id, state, reason, self.now_s())
        self.status_publisher.publish(message)

    def stop(self):
        """Abort active work and publish zero for orderly shutdown."""
        self.publish(*self.machine.shutdown())


def main(args=None):
    """Run the executor node."""
    # Keep the context valid until the final zero command is published.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = AgentExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
