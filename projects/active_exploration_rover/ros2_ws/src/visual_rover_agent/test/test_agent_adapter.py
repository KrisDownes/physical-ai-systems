"""Focused ROS adapter regression test."""

import json
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from visual_rover_agent.node import AgentExecutor


def test_commands_publish_status_motion_and_terminal_zero():
    """Exercise command transport through the real ROS adapter."""
    rclpy.init()
    agent = AgentExecutor()
    probe = Node('agent_adapter_test')
    commands = probe.create_publisher(String, '/agent_command', 10)
    odometry = probe.create_publisher(Odometry, '/odometry/filtered', 10)
    scans = probe.create_publisher(LaserScan, '/scan', 10)
    statuses = []
    velocities = []
    probe.create_subscription(
        String, '/agent_status', lambda msg: statuses.append(msg.data), 10)
    probe.create_subscription(
        Twist, '/cmd_vel', lambda msg: velocities.append(msg), 10)
    executor = SingleThreadedExecutor()
    executor.add_node(agent)
    executor.add_node(probe)

    scan = LaserScan()
    scan.angle_min = -3.14159
    scan.angle_increment = 6.28318 / 1079
    scan.range_min = 0.08
    scan.range_max = 10.0
    scan.ranges = [1.0] * 1080

    def spin_for(seconds, pose_x=0.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            pose = Odometry()
            pose.pose.pose.position.x = pose_x
            pose.pose.pose.orientation.w = 1.0
            odometry.publish(pose)
            scans.publish(scan)
            executor.spin_once(timeout_sec=0.01)

    try:
        spin_for(0.2)
        commands.publish(String(
            data='{"id":"adapter-stop","action":"stop"}'))
        spin_for(0.2)
        assert any(
            json.loads(status)['id'] == 'adapter-stop'
            for status in statuses
        )

        statuses.clear()
        velocities.clear()
        commands.publish(String(data=(
            '{"id":"adapter-drive","action":"drive",'
            '"distance_m":0.10}'
        )))
        spin_for(0.2)
        assert any(
            json.loads(status) == {
                'id': 'adapter-drive',
                'state': 'accepted',
                'reason': '',
                'sim_time_s': json.loads(status)['sim_time_s'],
            }
            for status in statuses
        )
        assert any(command.linear.x > 0.0 for command in velocities)

        spin_for(0.2, pose_x=0.09)
        assert any(
            json.loads(status)['state'] == 'succeeded'
            for status in statuses
        )
        assert velocities[-1].linear.x == 0.0
        assert velocities[-1].angular.z == 0.0
    finally:
        agent.stop()
        executor.remove_node(agent)
        executor.remove_node(probe)
        agent.destroy_node()
        probe.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
