import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        self.declare_parameter('forward_speed', 0.15)
        self.declare_parameter('maximum_turn_speed', 0.60)
        self.declare_parameter('heading_gain', 1.5)
        self.declare_parameter('lookahead_distance', 0.30)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter(
            'maximum_forward_heading_error_deg',
            35.0,
        )
        self.declare_parameter('path_timeout_s', 2.5)

        self.forward_speed = (
            self.get_parameter('forward_speed').value
        )
        self.maximum_turn_speed = (
            self.get_parameter('maximum_turn_speed').value
        )
        self.heading_gain = (
            self.get_parameter('heading_gain').value
        )
        self.lookahead_distance = (
            self.get_parameter('lookahead_distance').value
        )
        self.goal_tolerance = (
            self.get_parameter('goal_tolerance').value
        )
        self.maximum_forward_heading_error = math.radians(
            self.get_parameter(
                'maximum_forward_heading_error_deg'
            ).value
        )
        self.path_timeout_s = (
            self.get_parameter('path_timeout_s').value
        )

        self.latest_path = None
        self.latest_path_time = None

        self.path_subscription = self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10,
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel_raw',
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.control_timer = self.create_timer(
            0.1,  # 10 Hz
            self.control_callback,
        )

    def path_callback(self, path_message):
        # Store the newest path and its map frame.
        self.latest_path = path_message
        self.latest_path_time = self.get_clock().now()

    def control_callback(self):
        if (
            self.latest_path is None
            or self.latest_path_time is None
            or not self.latest_path.poses
        ):
            self.publish_cmd_vel(0.0, 0.0)
            return

        path_age = (
            self.get_clock().now()
            - self.latest_path_time
        )
        path_age_s = path_age.nanoseconds / 1_000_000_000

        if path_age_s > self.path_timeout_s:
            self.publish_cmd_vel(0.0, 0.0)
            return

        path_frame = self.latest_path.header.frame_id

        if not path_frame:
            self.publish_cmd_vel(0.0, 0.0)
            return

        try:
            robot_transform = self.tf_buffer.lookup_transform(
                path_frame,
                'base_footprint',
                Time(),
            )
        except TransformException:
            self.publish_cmd_vel(0.0, 0.0)
            return

        robot_translation = robot_transform.transform.translation
        robot_orientation = robot_transform.transform.rotation

        robot_x = robot_translation.x
        robot_y = robot_translation.y
        robot_yaw = self.quaternion_to_yaw(robot_orientation)

        goal_position = self.latest_path.poses[-1].pose.position

        goal_distance = math.hypot(
            goal_position.x - robot_x,
            goal_position.y - robot_y,
        )

        if goal_distance <= self.goal_tolerance:
            self.publish_cmd_vel(0.0, 0.0)
            return

        path_poses = self.latest_path.poses

        closest_index = min(
            range(len(path_poses)),
            key=lambda index: (
                (
                    path_poses[index].pose.position.x
                    - robot_x
                ) ** 2
                + (
                    path_poses[index].pose.position.y
                    - robot_y
                ) ** 2
            ),
        )

        lookahead_position = goal_position
        distance_along_path = 0.0

        previous_position = (
            path_poses[closest_index].pose.position
        )

        for path_pose in path_poses[closest_index + 1:]:
            current_position = path_pose.pose.position

            distance_along_path += math.hypot(
                current_position.x - previous_position.x,
                current_position.y - previous_position.y,
            )

            if distance_along_path >= self.lookahead_distance:
                lookahead_position = current_position
                break

            previous_position = current_position

        target_heading = math.atan2(
            lookahead_position.y - robot_y,
            lookahead_position.x - robot_x,
        )

        heading_error = self.normalize_angle(
            target_heading - robot_yaw
        )

        requested_turn_speed = (
            self.heading_gain * heading_error
        )

        angular_z = max(
            -self.maximum_turn_speed,
            min(
                self.maximum_turn_speed,
                requested_turn_speed,
            ),
        )

        forward_scale = max(
            0.0,
            1.0
            - abs(heading_error)
            / self.maximum_forward_heading_error,
        )

        linear_x = self.forward_speed * forward_scale

        self.publish_cmd_vel(linear_x, angular_z)

    def publish_cmd_vel(self, linear_x, angular_z):
        command = Twist()
        command.linear.x = linear_x
        command.angular.z = angular_z
        self.cmd_vel_publisher.publish(command)

    def quaternion_to_yaw(self, orientation):
        numerator = 2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        )
        denominator = 1.0 - 2.0 * (
            orientation.y ** 2
            + orientation.z ** 2
        )

        return math.atan2(numerator, denominator)

    def normalize_angle(self, angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )


def main(args=None):
    rclpy.init(args=args)

    node = PathFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
