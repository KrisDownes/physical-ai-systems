from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rover_exploration.frontier_detection import (
    cluster_frontier_cells,
    find_frontier_cells,
    grid_cell_center,
    representative_frontier_cell,
    world_point_to_grid_cell,
)
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')

        self.frontier_cells = set()
        self.frontier_clusters = []
        self.robot_grid_cell = None

        self.frontier_publisher = self.create_publisher(
            Marker,
            '/frontier_markers',
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
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

        try:
            map_time = Time.from_msg(map_message.header.stamp)
            robot_transform = self.tf_buffer.lookup_transform(
                map_message.header.frame_id,
                'base_footprint',
                map_time,
            )
        except TransformException as error:
            self.robot_grid_cell = None
            self.get_logger().warning(
                f'Could not find rover pose in map frame: {error}'
            )
        else:
            robot_x = robot_transform.transform.translation.x
            robot_y = robot_transform.transform.translation.y

            robot_row, robot_column = world_point_to_grid_cell(
                world_x=robot_x,
                world_y=robot_y,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            if (
                0 <= robot_row < map_message.info.height
                and 0 <= robot_column < map_message.info.width
            ):
                self.robot_grid_cell = (robot_row, robot_column)
            else:
                grid_x = (
                    (robot_x - map_message.info.origin.position.x)
                    / map_message.info.resolution
                )
                grid_y = (
                    (robot_y - map_message.info.origin.position.y)
                    / map_message.info.resolution
                )
                self.robot_grid_cell = None
                self.get_logger().warning(
                    f'Rover outside current map: '
                    f'world=({robot_x:.12f}, {robot_y:.12f}) '
                    f'origin=('
                    f'{map_message.info.origin.position.x:.12f}, '
                    f'{map_message.info.origin.position.y:.12f}) '
                    f'grid_float=({grid_y:.12f}, {grid_x:.12f}) '
                    f'grid=({robot_row}, {robot_column})'
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

        candidate_marker = Marker()
        candidate_marker.header = map_message.header
        candidate_marker.ns = 'frontier_candidates'
        candidate_marker.id = 0
        candidate_marker.type = Marker.SPHERE_LIST
        candidate_marker.action = Marker.ADD

        candidate_marker.pose.orientation.w = 1.0

        candidate_marker.scale.x = 0.15
        candidate_marker.scale.y = 0.15
        candidate_marker.scale.z = 0.15

        candidate_marker.color.r = 0.0
        candidate_marker.color.g = 1.0
        candidate_marker.color.b = 0.0
        candidate_marker.color.a = 1.0

        for cluster in self.frontier_clusters:
            representative_row, representative_column = (
                representative_frontier_cell(cluster)
            )
            candidate_x, candidate_y = grid_cell_center(
                row=representative_row,
                column=representative_column,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            candidate_point = Point()
            candidate_point.x = candidate_x
            candidate_point.y = candidate_y
            candidate_point.z = 0.10

            candidate_marker.points.append(candidate_point)

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
        self.frontier_publisher.publish(candidate_marker)

        self.get_logger().info(
            f'map={map_message.info.width}x{map_message.info.height} '
            f'frontier_cells={len(self.frontier_cells)} '
            f'frontier_clusters={len(self.frontier_clusters)} '
            f'robot_grid_cell={self.robot_grid_cell}'
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
