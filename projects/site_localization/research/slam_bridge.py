"""Sensor-only adapter for the installed official SLAM Toolbox mapper.

This node never subscribes to evaluator/truth topics. It supplies TF from wheel
odometry and the declared planar lidar calibration, republishes scans, and
exports the mapper's acquired map. Run in the mission's ROS domain.
"""
import argparse
import collections
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener
from tf2_ros import TransformException


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def atomic_text(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(value)
    pending.replace(path)


def export_map(message, out):
    """ROS corner origin to native AMCL input, with explicit state encoding.

    Native AMCL's text adapter cannot represent a rotated raster; reject one
    rather than silently moving mission coordinates. SLAM Toolbox grids are
    axis aligned. Intermediate probabilities are unknown, not assumed free.
    """
    info = message.info
    if abs(yaw(info.origin.orientation)) > 1e-6:
        raise ValueError('Native AMCL export requires an axis-aligned map origin')
    values = np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width)
    native = np.zeros_like(values)
    native[values == 0] = -1
    native[values == 100] = 1
    pending = out / 'map.txt.pending'
    with pending.open('w') as stream:
        stream.write(f'{info.width} {info.height} {info.resolution:.12g} '
                     f'{info.origin.position.x:.12g} {info.origin.position.y:.12g}\n')
        np.savetxt(stream, native, fmt='%d')
    pending.replace(out / 'map.txt')
    metadata = dict(source='official installed slam_toolbox /map occupancy grid',
                    frame=message.header.frame_id,
                    timestamp=stamp_seconds(message.header.stamp),
                    width=info.width, height=info.height, resolution=info.resolution,
                    origin=[info.origin.position.x, info.origin.position.y, 0.0],
                    native_states={'occupied': 1, 'free': -1, 'unknown': 0},
                    counts={str(v): int((native == v).sum()) for v in [-1, 0, 1]},
                    truth_used=False)
    atomic_text(out / 'map_metadata.json', json.dumps(metadata, indent=2) + '\n')


class MappingBridge(Node):
    def __init__(self, out):
        super().__init__('site_mapping_bridge')
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.broadcaster = TransformBroadcaster(self)
        self.static = StaticTransformBroadcaster(self)
        self.buffer = Buffer(cache_time=Duration(seconds=30))
        self.listener = TransformListener(self.buffer, self)
        self.scan_pub = self.create_publisher(LaserScan, '/mapping/scan', qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(PoseStamped, '/site/slam_tf_pose', 10)
        self.create_subscription(Odometry, '/site/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/site/scan', self.on_scan, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.create_subscription(PoseWithCovarianceStamped, '/pose', self.on_pose, 10)
        self.pending_odometry = collections.deque(maxlen=1000)
        self.latest_map = None
        self.odom_count = self.scan_count = self.map_count = self.tf_pose_count = 0
        self.files = {}
        self.writers = {}
        for name, header in {
            'mapper_pose': ['timestamp', 'x', 'y', 'yaw', 'cov_xx', 'cov_yy', 'cov_yawyaw'],
            'mapper_tf_pose': ['timestamp', 'x', 'y', 'yaw'],
            'wheel_odometry': ['timestamp', 'x', 'y', 'yaw'],
        }.items():
            self.files[name] = (out / f'{name}.csv').open('w')
            self.writers[name] = csv.writer(self.files[name])
            self.writers[name].writerow(header)
        self.frames = (out / 'sensor_frames.txt').open('w')
        self.frames.write('0 0 0 8\n')
        self.odom_history = collections.deque(maxlen=2000)
        self.scan_queue = collections.deque(maxlen=100)
        calibration = TransformStamped()
        calibration.header.stamp = self.get_clock().now().to_msg()
        calibration.header.frame_id = 'base_link'
        calibration.child_frame_id = 'laser'
        # Robot in sim/generate.py: lidar is at x=y=yaw=0 relative to base.
        # Its physical height is irrelevant to this planar mapper, not to 3D use.
        calibration.transform.rotation.w = 1.0
        self.static.sendTransform(calibration)
        self.timer = self.create_timer(.1, self.drain_tf)
        atomic_text(out / 'input_contract.json', json.dumps(dict(
            subscribed_topics=['/site/odom', '/site/scan', '/map', '/pose', '/tf', '/tf_static', '/clock'],
            method_inputs=['wheel odometry', '2D lidar', 'declared planar calibration'],
            calibration_base_from_laser_xyyaw=[0, 0, 0], truth_used=False,
            map_frame='map', odom_frame='odom', base_frame='base_link', laser_frame='laser',
            first_pose_assumption='SLAM starts in odometry frame; dock-to-mission alignment is external',
            tf_pose_policy='TF interpolated at the exact wheel-odometry timestamp; no latest-time substitute',
            sensor_frames='Timestamp-synchronized scan/wheel rows; native AMCL format; no truth',
        ), indent=2) + '\n')

    def write_row(self, name, row):
        self.writers[name].writerow(row)
        self.files[name].flush()

    def on_odom(self, message):
        pose = message.pose.pose
        transform = TransformStamped()
        transform.header = copy.deepcopy(message.header)
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.rotation = copy.deepcopy(pose.orientation)
        self.broadcaster.sendTransform(transform)
        self.pending_odometry.append((message.header.stamp, pose))
        timestamp = stamp_seconds(message.header.stamp)
        self.odom_history.append((timestamp, np.array([pose.position.x, pose.position.y, yaw(pose.orientation)])))
        self.write_row('wheel_odometry', [timestamp, pose.position.x, pose.position.y, yaw(pose.orientation)])
        self.odom_count += 1
        self.drain_scans()

    def on_scan(self, message):
        scan = copy.deepcopy(message)
        scan.header.frame_id = 'laser'
        self.scan_pub.publish(scan)
        self.scan_queue.append(message)
        self.scan_count += 1
        self.drain_scans()

    def drain_scans(self):
        while self.scan_queue and len(self.odom_history) >= 2:
            message = self.scan_queue[0]
            timestamp = stamp_seconds(message.header.stamp)
            if timestamp > self.odom_history[-1][0]:
                break
            self.scan_queue.popleft()
            if timestamp < self.odom_history[0][0]:
                continue
            times = np.array([entry[0] for entry in self.odom_history])
            hi = min(int(np.searchsorted(times, timestamp, side='right')), len(times) - 1)
            lo = max(0, hi - 1)
            if times[hi] - times[lo] > .15:
                continue
            first, second = self.odom_history[lo][1], self.odom_history[hi][1]
            delta = second - first
            delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
            pose = first + (timestamp - times[lo]) / max(1e-12, times[hi] - times[lo]) * delta
            ranges = np.asarray(message.ranges, dtype=float)
            ranges[(~np.isfinite(ranges)) | (ranges < message.range_min) | (ranges >= 8)] = 8
            angles = message.angle_min + np.arange(len(ranges)) * message.angle_increment
            row = [timestamp, *pose, len(ranges), *np.ravel(np.c_[ranges, angles])]
            self.frames.write(' '.join(map(str, row)) + '\n')
            self.frames.flush()

    def on_map(self, message):
        try:
            export_map(message, self.out)
        except ValueError as error:
            self.get_logger().error(str(error))
            atomic_text(self.out / 'export_error.txt', str(error) + '\n')
            return
        self.latest_map = message
        self.map_count += 1

    def on_pose(self, message):
        p = message.pose.pose
        cov = message.pose.covariance
        self.write_row('mapper_pose', [stamp_seconds(message.header.stamp), p.position.x,
                                      p.position.y, yaw(p.orientation), cov[0], cov[7], cov[35]])

    def drain_tf(self):
        # map->odom may arrive after the wheel message. Wait in a bounded queue;
        # never substitute a later transform and silently alter trajectory error.
        remaining = collections.deque(maxlen=1000)
        for stamp, pose in self.pending_odometry:
            try:
                transform = self.buffer.lookup_transform('map', 'odom', Time.from_msg(stamp))
            except TransformException:
                if self.get_clock().now().nanoseconds * 1e-9 - stamp_seconds(stamp) < 10:
                    remaining.append((stamp, pose))
                continue
            t = transform.transform
            angle = yaw(t.rotation)
            x = math.cos(angle) * pose.position.x - math.sin(angle) * pose.position.y + t.translation.x
            y = math.sin(angle) * pose.position.x + math.cos(angle) * pose.position.y + t.translation.y
            heading = angle + yaw(pose.orientation)
            heading = math.atan2(math.sin(heading), math.cos(heading))
            self.write_row('mapper_tf_pose', [stamp_seconds(stamp), x, y, heading])
            message = PoseStamped()
            message.header.stamp = stamp
            message.header.frame_id = 'map'
            message.pose.position.x = x
            message.pose.position.y = y
            message.pose.orientation.z = math.sin(heading/2)
            message.pose.orientation.w = math.cos(heading/2)
            self.pose_pub.publish(message)
            self.tf_pose_count += 1
        self.pending_odometry = remaining

    def close(self):
        if self.latest_map is not None:
            export_map(self.latest_map, self.out)
        for stream in self.files.values():
            stream.close()
        self.frames.close()
        atomic_text(self.out / 'bridge_summary.json', json.dumps(dict(
            wheel_messages=self.odom_count, scan_messages=self.scan_count,
            map_messages=self.map_count, matched_tf_poses=self.tf_pose_count,
            pending_tf_poses=len(self.pending_odometry), map_exported=self.latest_map is not None,
        ), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = MappingBridge(args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
