import math

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from rover_exploration.exploration_policy import ExplorationPolicy, PolicyConfig
from rover_exploration.frontier_selection import (
    cluster_frontier_cells,
    find_frontier_cells,
    grid_cell_center,
    world_point_to_grid_cell,
)
from rover_exploration.grid_planning import (
    build_planning_grid,
    close_occupied_walls,
    inflate_occupancy_grid,
    pad_unknown_space,
)
from rover_exploration.result_contract import serialize_result
from rover_exploration.stuck_detection import quaternion_yaw
from std_msgs.msg import Bool, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker


class FrontierDetector(Node):
    """Connect map/TF/recovery topics to one exploration policy."""

    def __init__(self):
        super().__init__('frontier_detector')

        policy_config = PolicyConfig(
            goal_reached_distance_m=self._parameter(
                'goal_reached_distance_m', 0.25
            ),
            maximum_goal_path_failures=self._parameter(
                'maximum_goal_path_failures', 3
            ),
            maximum_fresh_approaches_per_target=self._parameter(
                'maximum_fresh_approaches_per_target', 3
            ),
            stuck_window_s=self._parameter('stuck.window_s', 6.0),
            stuck_progress_threshold_m=self._parameter(
                'stuck.progress_threshold_m', 0.05
            ),
            stuck_alignment_threshold_rad=self._parameter(
                'stuck.alignment_threshold_rad', 0.39269908169872414
            ),
            blacklist_radius_m=self._parameter('blacklist.radius_m', 0.75),
            blacklist_duration_s=self._parameter(
                'blacklist.duration_s', 30.0
            ),
            permanent_after_failures=self._parameter(
                'blacklist.permanent_after_failures', 2
            ),
            visited_radius_m=self._parameter('visited.radius_m', 0.60),
            permanent_exclusion_radius_m=self._parameter(
                'permanent_exclusion_radius_m', 0.20
            ),
            distance_slack_m=self._parameter(
                'selection.distance_slack_m', 2.0
            ),
            completion_debounce_period_s=self._parameter(
                'completion_debounce.period_s', 8.0
            ),
            approach_search_radius_m=self._parameter(
                'selection.approach_search_radius_m', 1.5
            ),
        )
        self.wall_closing_radius_m = self._parameter(
            'planning.wall_closing_radius_m', 0.05
        )
        self.unknown_clearance_m = self._parameter(
            'planning.unknown_clearance_m', 0.10
        )
        self.policy = ExplorationPolicy(policy_config)
        self.latest_pose = None
        self.terminal_published = False

        self.rover_length_m = 0.45
        self.rover_width_m = 0.30
        self.path_clearance_m = 0.05

        self.frontier_publisher = self.create_publisher(
            Marker, '/frontier_markers', 10
        )
        self.planning_grid_publisher = self.create_publisher(
            OccupancyGrid, '/planning_grid', 10
        )
        self.path_publisher = self.create_publisher(Path, '/planned_path', 10)
        self.recovery_request_publisher = self.create_publisher(
            Bool, '/recovery_request', 10
        )

        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.exploration_complete_publisher = self.create_publisher(
            Bool, '/exploration_complete', state_qos
        )
        self.exploration_result_publisher = self.create_publisher(
            String, '/exploration_result', state_qos
        )
        self.recovery_status_subscription = self.create_subscription(
            Bool, '/recovery_status', self.recovery_status_callback, state_qos
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_subscription = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.pose_timer = self.create_timer(0.5, self.pose_timer_callback)
        self.stuck_timer = self.create_timer(1.0, self.stuck_check_callback)

        initial = Bool()
        initial.data = False
        self.exploration_complete_publisher.publish(initial)

    def _parameter(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def node_time_s(self):
        return self.get_clock().now().nanoseconds / 1_000_000_000

    def map_callback(self, map_message):
        """Plan once from the newest map and publish its complete output."""
        if self.policy.complete:
            self._publish_path(map_message.header, map_message.info, None)
            return

        frontier_cells = find_frontier_cells(
            map_message.data,
            map_message.info.width,
            map_message.info.height,
        )
        clusters = cluster_frontier_cells(frontier_cells, min_cluster_size=5)
        robot_world, robot_cell = self._map_pose(map_message)
        planning_data, inflation_cells = self._build_planning_grid(map_message)

        update = self.policy.update(
            raw_data=map_message.data,
            planning_data=planning_data,
            width=map_message.info.width,
            height=map_message.info.height,
            resolution=map_message.info.resolution,
            origin_x=map_message.info.origin.position.x,
            origin_y=map_message.info.origin.position.y,
            frontier_cells=frontier_cells,
            frontier_clusters=clusters,
            robot_cell=robot_cell,
            robot_world=robot_world,
            now_s=self.node_time_s(),
        )

        self._publish_grid(map_message, planning_data)
        self._publish_path(map_message.header, map_message.info, update.path)
        retained_frontier_cells = set().union(*clusters) if clusters else set()
        self._publish_markers(map_message, retained_frontier_cells, update)
        self._log_cycle(
            map_message, clusters, robot_cell, update, inflation_cells
        )
        self._log_transition(update)

        # Stop the path follower before the terminal state becomes visible.
        if update.completed_now:
            self._publish_terminal()

    def _map_pose(self, map_message):
        try:
            transform = self.tf_buffer.lookup_transform(
                map_message.header.frame_id,
                'base_footprint',
                Time.from_msg(map_message.header.stamp),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Could not find rover pose in map frame: {error}'
            )
            return None, None

        x = transform.transform.translation.x
        y = transform.transform.translation.y
        cell = world_point_to_grid_cell(
            x,
            y,
            map_message.info.resolution,
            map_message.info.origin.position.x,
            map_message.info.origin.position.y,
        )
        if not (
            0 <= cell[0] < map_message.info.height
            and 0 <= cell[1] < map_message.info.width
        ):
            self.get_logger().warning(
                f'Rover outside current map: world=({x:.3f}, {y:.3f}) '
                f'grid={cell}'
            )
            return (x, y), None
        return (x, y), cell

    def _build_planning_grid(self, map_message):
        info = map_message.info
        rover_radius = math.hypot(
            self.rover_length_m / 2.0, self.rover_width_m / 2.0
        )
        inflation_cells = math.ceil(
            (rover_radius + self.path_clearance_m) / info.resolution
        )
        inflated = inflate_occupancy_grid(
            map_message.data, info.width, info.height, inflation_cells
        )
        conditioned = close_occupied_walls(
            map_message.data,
            info.width,
            info.height,
            closing_radius_cells=max(
                0, int(round(self.wall_closing_radius_m / info.resolution))
            ),
        )
        conditioned = pad_unknown_space(
            conditioned,
            info.width,
            info.height,
            padding_radius_cells=max(
                0, int(round(self.unknown_clearance_m / info.resolution))
            ),
        )
        return (
            build_planning_grid(map_message.data, inflated, conditioned),
            inflation_cells,
        )

    def _publish_grid(self, map_message, planning_data):
        message = OccupancyGrid()
        message.header = map_message.header
        message.info = map_message.info
        message.data = [int(value) for value in planning_data]
        self.planning_grid_publisher.publish(message)

    def _publish_path(self, header, info, path):
        message = Path()
        message.header = header
        for row, column in path or ():
            x, y = grid_cell_center(
                row,
                column,
                info.resolution,
                info.origin.position.x,
                info.origin.position.y,
            )
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.path_publisher.publish(message)

    @staticmethod
    def _marker(header, namespace, marker_type, color, scale):
        marker = Marker()
        marker.header = header
        marker.ns = namespace
        marker.id = 0
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        return marker

    @staticmethod
    def _point(x, y, z):
        point = Point()
        point.x, point.y, point.z = x, y, z
        return point

    def _publish_markers(self, map_message, frontier_cells, update):
        info = map_message.info
        frontiers = self._marker(
            map_message.header,
            'frontiers',
            Marker.POINTS,
            (1.0, 0.0, 0.0, 1.0),
            (info.resolution, info.resolution, 0.0),
        )
        for cell in frontier_cells:
            x, y = grid_cell_center(
                cell[0], cell[1], info.resolution,
                info.origin.position.x, info.origin.position.y,
            )
            frontiers.points.append(self._point(x, y, 0.05))

        for marker in (
            frontiers,
            self._candidate_marker(map_message, update),
            self._selected_marker(map_message, update.selected_cell),
            self._path_marker(map_message, update.path),
        ):
            self.frontier_publisher.publish(marker)
        for marker in self._memory_markers(map_message):
            self.frontier_publisher.publish(marker)

    def _candidate_marker(self, map_message, update):
        marker = self._marker(
            map_message.header,
            'frontier_candidates',
            Marker.SPHERE_LIST,
            (0.0, 1.0, 0.0, 1.0),
            (0.15, 0.15, 0.15),
        )
        info = map_message.info
        for cell in update.stats.approach_cells:
            x, y = grid_cell_center(
                cell[0], cell[1], info.resolution,
                info.origin.position.x, info.origin.position.y,
            )
            marker.points.append(self._point(x, y, 0.10))
        return marker

    def _selected_marker(self, map_message, cell):
        marker = self._marker(
            map_message.header,
            'selected_frontier',
            Marker.SPHERE,
            (0.0, 0.0, 1.0, 1.0),
            (0.25, 0.25, 0.25),
        )
        if cell is None:
            marker.action = Marker.DELETE
            return marker
        info = map_message.info
        x, y = grid_cell_center(
            cell[0], cell[1], info.resolution,
            info.origin.position.x, info.origin.position.y,
        )
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.15
        return marker

    def _path_marker(self, map_message, path):
        marker = self._marker(
            map_message.header,
            'planned_path',
            Marker.LINE_STRIP,
            (1.0, 1.0, 0.0, 1.0),
            (0.04, 0.0, 0.0),
        )
        if path is None:
            marker.action = Marker.DELETE
            return marker
        info = map_message.info
        for cell in path:
            x, y = grid_cell_center(
                cell[0], cell[1], info.resolution,
                info.origin.position.x, info.origin.position.y,
            )
            marker.points.append(self._point(x, y, 0.20))
        return marker

    def _memory_markers(self, map_message):
        memory = self.policy.memory
        groups = (
            ('temp_failed', memory.active_cooldowns(self.node_time_s()),
             (1.0, 0.55, 0.0, 0.35),
             self.policy.config.blacklist_radius_m),
            ('permanent_failed', memory.permanent_failures,
             (0.8, 0.0, 0.8, 0.35),
             self.policy.config.permanent_exclusion_radius_m),
            ('visited', memory.visited, (0.45, 0.55, 0.45, 0.35),
             self.policy.config.visited_radius_m),
        )
        markers = []
        for namespace, regions, color, radius in groups:
            marker = self._marker(
                map_message.header,
                namespace,
                Marker.SPHERE_LIST,
                color,
                (radius * 2.0, radius * 2.0, 0.05),
            )
            for region in regions:
                if hasattr(region, 'x'):
                    x, y = region.x, region.y
                else:
                    x, y = region
                marker.points.append(self._point(x, y, 0.02))
            markers.append(marker)
        return markers

    def recovery_status_callback(self, status):
        if self.policy.recovery_status(status.data):
            self.get_logger().info(
                'Recovery cycle ended; stuck detection window reset'
            )

    def pose_timer_callback(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', Time()
            )
        except TransformException:
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self.latest_pose = (
            translation.x,
            translation.y,
            quaternion_yaw(rotation.x, rotation.y, rotation.z, rotation.w),
        )

    def stuck_check_callback(self):
        if self.latest_pose is None:
            return
        event = self.policy.observe_pose(self.node_time_s(), self.latest_pose)
        if event is None:
            return
        request = Bool()
        request.data = True
        self.recovery_request_publisher.publish(request)
        self._log_failure(event.failure_outcome, event.goal_x, event.goal_y)
        self.get_logger().warning(
            f'Rover made no progress toward goal '
            f'({event.goal_x:.3f}, {event.goal_y:.3f}); recovery requested'
        )

    def _publish_terminal(self):
        if self.terminal_published:
            return
        completion_time_s = self.node_time_s()
        result = String()
        result.data = serialize_result(
            **self.policy.result_values(completion_time_s)
        )
        state = Bool()
        state.data = True
        self.terminal_published = True
        self.exploration_complete_publisher.publish(state)
        self.exploration_result_publisher.publish(result)
        self.get_logger().warning(
            f'Exploration terminal: {self.policy.terminal.outcome}; {result.data}'
        )

    def _log_transition(self, update):
        if update.failure is not None:
            self._log_failure(*update.failure)
        if update.goal_reached is not None:
            self.get_logger().info(
                f'Goal reached at {update.goal_reached}; region marked visited'
            )
        if update.goal_assigned is not None:
            self.get_logger().info(f'Goal assigned at {update.goal_assigned}')
        if update.cooldown_hold_started:
            self.get_logger().warning(
                'Completion deferred by active temporary cooldown; holding'
            )
        if update.debounce_started:
            self.get_logger().warning(
                'No selectable goal; debouncing terminal decision'
            )

    def _log_failure(self, outcome, goal_x, goal_y):
        detail = (
            'promoted to permanent exclusion'
            if outcome == 'promoted'
            else 'temporary cooldown recorded'
        )
        self.get_logger().warning(
            f'Goal region ({goal_x:.2f}, {goal_y:.2f}) {detail}'
        )

    def _log_cycle(self, map_message, clusters, robot_cell, update, inflation):
        stats = update.stats
        self.get_logger().info(
            f'map={map_message.info.width}x{map_message.info.height} '
            f'frontier_clusters={len(clusters)} robot_grid_cell={robot_cell} '
            f'selected={update.selected_cell} path_cells={len(update.path or ())} '
            f'inflation_radius_cells={inflation} '
            f'approaches={len(stats.approach_cells)} '
            f'temporary_rejected={stats.temporary_rejected} '
            f'permanent_rejected={stats.permanent_rejected} '
            f'visited_rejected={stats.visited_rejected} '
            f'retry_exhausted={stats.retry_exhausted} '
            f'unreachable_clusters={stats.unreachable_clusters} '
            f'eligible={stats.eligible} recovery={self.policy.recovery_state}'
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
