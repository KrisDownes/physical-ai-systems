"""Thin ROS adapter for the bounded action state machine."""

import math
import json
import time
from rclpy.clock import Clock, ClockType

from geometry_msgs.msg import Twist

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan

from std_msgs.msg import String

from visual_rover_agent.parser import (
    CommandError,
    parse_command,
    serialize_status,
)
from visual_rover_agent.state_machine import ActionMachine, Limits, Event


def sector_minimum(scan, center, half_width):
    """Find the nearest valid return in an angular sector."""
    if (not all(math.isfinite(v) for v in (
            scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max))
            or scan.angle_increment <= 0 or scan.range_min < 0
            or scan.range_max <= scan.range_min):
        return None
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
            'command_timeout_s': 10.0,  # ROS/simulation seconds
            'clock_stall_timeout_s': 1.0,  # monotonic wall seconds without ROS progress
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
        self.clock_stall_timeout_s = values.pop('clock_stall_timeout_s')
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
        self.timing = self.create_publisher(String, "/agent_timing", 100)
        self.command_id = None
        self.wall_deadline = None
        self.last_control_sim_s = None
        self.create_timer(0.05, self.control_callback, clock=Clock(clock_type=ClockType.STEADY_TIME))

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
        self.command_id = command.command_id
        self.event("executor_received")
        events, velocity = self.machine.submit(command, self.now_s())
        self.wall_deadline = time.monotonic() + self.clock_stall_timeout_s
        self.last_control_sim_s = self.now_s()
        for event in events:
            if event.state == "accepted":
                self.event("executor_accepted", id=event.command_id)
        self.publish(events, velocity)

    def odom_callback(self, message):
        """Forward planar odometry to the machine."""
        p = message.pose.pose.position
        q = message.pose.pose.orientation
        components = (q.x, q.y, q.z, q.w)
        if (not all(math.isfinite(v) for v in components)
                or abs(sum(v * v for v in components) - 1.0) > 0.01):
            self.machine.odom_time = None
            return
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        stamp = message.header.stamp
        self.machine.update_odometry(p.x, p.y, yaw, stamp.sec + stamp.nanosec / 1e9)

    def scan_callback(self, message):
        """Reduce the 360-degree scan to safety clearances."""
        front = sector_minimum(message, 0.0, math.radians(25.0))
        rear = sector_minimum(message, math.pi, math.radians(30.0))
        all_around = sector_minimum(message, 0.0, math.pi)
        stamp = message.header.stamp
        self.machine.update_scan(front, rear, all_around, stamp.sec + stamp.nanosec / 1e9)

    def control_callback(self):
        """Advance control without sleeps using ROS time."""
        self.event("control_tick")
        sim_now, wall_now = self.now_s(), time.monotonic()
        # Slow simulation is not a stalled clock. The machine retains its
        # independent 10 ROS-second action deadline and sensor safety checks.
        if self.last_control_sim_s is None or sim_now > self.last_control_sim_s:
            self.wall_deadline = wall_now + self.clock_stall_timeout_s
        self.last_control_sim_s = sim_now
        if self.machine.active and self.wall_deadline is not None and wall_now > self.wall_deadline:
            events, velocity = self.machine.shutdown()
            self.publish([Event(e.command_id, e.state, 'clock_stalled') for e in events], velocity)
        else:
            self.publish(*self.machine.tick(sim_now))

    def publish(self, events, velocity):
        """Publish velocity before any resulting status transitions."""
        self.publish_velocity(velocity)
        for event in events:
            self.publish_status(event.command_id, event.state, event.reason)

    def event(self, kind, **data):
        self.timing.publish(String(data=json.dumps(dict(kind=kind, wall_s=time.monotonic(), sim_s=self.now_s(), **({"id": self.command_id} | data)))))

    def publish_velocity(self, velocity):
        """Publish the requested planar velocity."""
        message = Twist()
        message.linear.x, message.angular.z = velocity
        self.velocity_publisher.publish(message)
        self.event("velocity_published", velocity=list(velocity))

    def publish_status(self, command_id, state, reason):
        """Publish one strict JSON status."""
        message = String()
        message.data = serialize_status(
            command_id, state, reason, self.now_s())
        self.status_publisher.publish(message)
        self.event("executor_status", id=command_id, state=state, reason=reason)

    def stop(self):
        """Abort active work and publish zero for orderly shutdown."""
        self.publish(*self.machine.shutdown())


def main(args=None):
    """Run the executor node."""
    rclpy.init(args=args)
    node = AgentExecutor()
    node.context.on_shutdown(node.stop)
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
