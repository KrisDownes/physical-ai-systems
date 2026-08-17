from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rover_exploration.frontier_detection import (
    find_frontier_cells,
    grid_cell_center,
)
from visualization_msgs.msg import Marker


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')

        self.frontier_cells = set()

        self.frontier_publisher = self.create_publisher(
            Marker,
            '/frontier_markers',
            10,
        )

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

        marker = Marker()
        marker.header = map_message.header
        marker.ns = 'frontiers'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = map_message.info.resolution
        marker.scale.y = map_message.info.resolution

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for row, column in self.frontier_cells:
            world_x, world_y = grid_cell_center(
                row=row,
                column=column,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            point = Point()
            point.x = world_x
            point.y = world_y
            point.z = 0.05

            marker.points.append(point)

        self.frontier_publisher.publish(marker)

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
