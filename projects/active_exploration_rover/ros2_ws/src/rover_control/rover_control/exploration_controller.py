import math

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rover_control.scan_processing import nearest_valid_range_in_sector
from sensor_msgs.msg import LaserScan


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        self.forward_speed = 0.20
        self.turn_trigger_distance = 0.80
        self.turn_speed = 0.60

        self.front_distance = None
        self.left_distance = None
        self.right_distance = None

        self.front_sector_half_width = math.radians(20)
        self.side_sector_center = math.radians(60)
        self.side_sector_half_width = math.radians(30)

        self.command_publisher = self.create_publisher(
            Twist,
            '/cmd_vel_raw',
            10,
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(
            0.1,
            self.control_callback,
        )

    def control_callback(self):
        command = Twist()
        if self.front_distance is None:
            pass
        elif self.front_distance > self.turn_trigger_distance:
            command.linear.x = self.forward_speed
        else:
            if self.left_distance is None and self.right_distance is None:
                pass
            elif self.right_distance is None:
                command.angular.z = self.turn_speed
            elif self.left_distance is None:
                command.angular.z = -self.turn_speed
            elif self.left_distance >= self.right_distance:
                command.angular.z = self.turn_speed
            else:
                command.angular.z = -self.turn_speed

        self.command_publisher.publish(command)

    def scan_callback(self, scan):

        self.front_distance = nearest_valid_range_in_sector(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            sector_center=0.0,
            sector_half_width=self.front_sector_half_width,
        )

        self.left_distance = nearest_valid_range_in_sector(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            sector_center=self.side_sector_center,
            sector_half_width=self.side_sector_half_width,
        )

        self.right_distance = nearest_valid_range_in_sector(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            sector_center=-self.side_sector_center,
            sector_half_width=self.side_sector_half_width,
        )


def main(args=None):
    rclpy.init(args=args)

    node = ExplorationController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
