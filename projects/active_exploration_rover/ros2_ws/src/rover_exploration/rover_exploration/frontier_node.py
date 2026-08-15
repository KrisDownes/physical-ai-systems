from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node

from rover_exploration.frontier_detection import find_frontier_cells


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')

        self.frontier_cells = set()

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10,
        )

    def map_callback(self, map_message):
        self.frontier_cells = find_frontier_cells(
            data=map_message.data,
            width=map_message.info.width,
            height=map_message.info.height,
        )

        self.get_logger().info(
            f'map={map_message.info.width}x{map_message.info.height} '
            f'frontier_cells={len(self.frontier_cells)}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
