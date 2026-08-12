import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rover_control.safety_logic import update_front_blocked_state
from rover_control.scan_processing import nearest_valid_range_in_sector
from sensor_msgs.msg import LaserScan


class ObstacleGuard(Node):
    def __init__(self):
        super().__init__('obstacle_guard')

        self.stop_distance = 0.45
        self.resume_distance = 0.55
        self.sector_half_width = math.radians(20)

        self.front_blocked = False
        self.scan_is_stale = True
        self.last_scan_time = None
        self.scan_timeout_s = 0.5

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
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

        self.scan_watchdog_timer = self.create_timer(
            0.1,
            self.scan_watchdog_callback,
        )

    def scan_callback(self, scan):
        self.last_scan_time = self.get_clock().now()
        self.scan_is_stale = False

        nearest_distance = nearest_valid_range_in_sector(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            sector_center=0.0,
            sector_half_width=self.sector_half_width,
        )

        was_blocked = self.front_blocked

        self.front_blocked = update_front_blocked_state(
            nearest_distance=nearest_distance,
            was_blocked=was_blocked,
            stop_distance=self.stop_distance,
            resume_distance=self.resume_distance,
        )

        if self.front_blocked and not was_blocked:
            self.command_publisher.publish(Twist())

    def command_callback(self, command):
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

    def scan_watchdog_callback(self):
        if self.last_scan_time is None:
            return

        scan_age = self.get_clock().now() - self.last_scan_time
        scan_age_s = scan_age.nanoseconds / 1_000_000_000

        if scan_age_s > self.scan_timeout_s and not self.scan_is_stale:
            self.scan_is_stale = True
            self.command_publisher.publish(Twist())


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
