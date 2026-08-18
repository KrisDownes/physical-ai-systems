from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rover_exploration.frontier_detection import (
    cluster_frontier_cells,
    find_frontier_cells,
    grid_cell_center,
    representative_frontier_cell,
)
from visualization_msgs.msg import Marker


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')

        self.frontier_cells = set()
        self.frontier_clusters = []

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

        self.frontier_clusters = cluster_frontier_cells(
            self.frontier_cells,
            min_cluster_size=5,
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

        centroid_marker = Marker()
        centroid_marker.header = map_message.header
        centroid_marker.ns = 'frontier_candidates'
        centroid_marker.id = 0
        centroid_marker.type = Marker.SPHERE_LIST
        centroid_marker.action = Marker.ADD

        centroid_marker.pose.orientation.w = 1.0

        centroid_marker.scale.x = 0.15
        centroid_marker.scale.y = 0.15
        centroid_marker.scale.z = 0.15

        centroid_marker.color.r = 0.0
        centroid_marker.color.g = 1.0
        centroid_marker.color.b = 0.0
        centroid_marker.color.a = 1.0

        for cluster in self.frontier_clusters:
            representative_row, representative_column = (
                representative_frontier_cell(cluster)
            )
            centroid_x, centroid_y = grid_cell_center(
                row=representative_row,
                column=representative_column,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            centroid_point = Point()
            centroid_point.x = centroid_x
            centroid_point.y = centroid_y
            centroid_point.z = 0.10

            centroid_marker.points.append(centroid_point)

            for row, column in cluster:
                world_x, world_y = grid_cell_center(
                    row=row,
                    column=column,
                    resolution=map_message.info.resolution,
                    origin_x=map_message.info.origin.position.x,
                    origin_y=map_message.info.origin.position.y,
                )

                frontier_point = Point()
                frontier_point.x = world_x
                frontier_point.y = world_y
                frontier_point.z = 0.05

                marker.points.append(frontier_point)

        self.frontier_publisher.publish(marker)
        self.frontier_publisher.publish(centroid_marker)

        self.get_logger().info(
            f'map={map_message.info.width}x{map_message.info.height} '
            f'frontier_cells={len(self.frontier_cells)} '
            f'frontier_clusters={len(self.frontier_clusters)}'
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
