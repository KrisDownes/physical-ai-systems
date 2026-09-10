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
            pose.header.stamp = probe.get_clock().now().to_msg()
            scan.header.stamp = pose.header.stamp
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


def test_old_sensor_header_is_not_refreshed_by_receipt():
    from visual_rover_agent.state_machine import ActionMachine, Limits
    from visual_rover_agent.parser import parse_command
    agent = AgentExecutor.__new__(AgentExecutor)
    agent.machine = ActionMachine(Limits())
    pose = Odometry()
    pose.pose.pose.orientation.w = 1.0
    pose.header.stamp.nanosec = 100_000_000
    agent.odom_callback(pose)
    agent.machine.update_scan(1., 1., 1., 1.)
    events, velocity = agent.machine.submit(parse_command(
        '{"id":"old","action":"drive","distance_m":0.1}', .5, 90), 1.)
    assert events[0].reason == 'stale_odometry'
    assert velocity == (0., 0.)


def test_invalid_scan_geometry_is_unavailable():
    from visual_rover_agent.node import sector_minimum
    scan = LaserScan()
    scan.angle_min = float('inf')
    scan.angle_increment = .1
    scan.range_min, scan.range_max = .1, 10.
    scan.ranges = [1.]
    assert sector_minimum(scan, 0., 1.) is None



def test_slow_simulation_and_stalled_clock_have_different_deadlines(monkeypatch):
    from visual_rover_agent import node
    from visual_rover_agent.state_machine import ActionMachine, Limits
    from visual_rover_agent.parser import parse_command
    agent=AgentExecutor.__new__(AgentExecutor)
    agent.machine=ActionMachine(Limits())
    agent.clock_stall_timeout_s=1.0
    agent.wall_deadline=1.0
    agent.last_control_sim_s=0.0
    agent.event=lambda *a,**k:None
    published=[]
    agent.publish=lambda events,velocity:published.append((events,velocity))
    agent.machine.update_odometry(0.,0.,0.,0.)
    agent.machine.update_scan(2.,2.,2.,0.)
    agent.machine.submit(parse_command('{"id":"slow","action":"drive","distance_m":0.5}',.5,90),0.)
    clock=[0.,0.]
    agent.now_s=lambda:clock[0]
    monkeypatch.setattr(node.time,'monotonic',lambda:clock[1])
    # Feedback continues at RTF .52 beyond the old 10-wall-second cutoff.
    for step in range(1,121):
        clock[:]=[step*.052,step*.1]
        agent.machine.update_odometry(.4,0.,0.,clock[0])
        agent.machine.update_scan(2.,2.,2.,clock[0])
        agent.control_callback()
    assert agent.machine.active is not None
    agent.machine.update_odometry(.49,0.,0.,clock[0])
    agent.control_callback()
    assert published[-1][0][0].state=='succeeded'
    # A paused clock must still cause a wall-clock stop, with an honest reason.
    agent.machine.submit(parse_command('{"id":"paused","action":"drive","distance_m":0.1}',.5,90),clock[0])
    clock[1]+=1.1
    agent.control_callback()
    assert published[-1][0][0].reason=='clock_stalled'
    assert published[-1][1]==(0.,0.)
