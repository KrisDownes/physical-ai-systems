from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        self.forward_speed = 0.20

        self.command_publisher = self.create_publisher(
            Twist,
            '/cmd_vel_raw',
            10,
        )

        self.control_timer = self.create_timer(
            0.1,
            self.control_callback,
        )

    def control_callback(self):
        command = Twist()

        command.linear.x = self.forward_speed

        self.command_publisher.publish(command)


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
