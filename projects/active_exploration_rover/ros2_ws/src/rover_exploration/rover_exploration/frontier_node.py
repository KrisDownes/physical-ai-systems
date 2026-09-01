from collections import deque
import json
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
from rover_exploration.frontier_detection import (
    cluster_frontier_cells,
    find_frontier_cells,
    grid_cell_center,
    world_point_to_grid_cell,
)
from rover_exploration.frontier_memory import (
    is_excluded,
    prune_expired_cooldowns,
    record_failure,
)
from rover_exploration.grid_planning import (
    build_planning_grid,
    close_occupied_walls,
    compute_reachable_component,
    find_cluster_approach_cell_reachable,
    find_escape_path,
    inflate_occupancy_grid,
    is_traversable_grid_cell,
    pad_unknown_space,
    reconstruct_grid_path,
    select_cluster_weighted_goal,
)
from rover_exploration.recovery_coordination import (
    RecoveryCoordinationState,
)
from rover_exploration.stuck_detection import (
    is_stuck,
    quaternion_yaw,
)
from std_msgs.msg import Bool, String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker


class FrontierDetector(Node):
    def __init__(self):
        super().__init__('frontier_detector')

        self.declare_parameter('goal_reached_distance_m', 0.25)
        self.declare_parameter('maximum_goal_path_failures', 3)
        self.declare_parameter(
            'maximum_fresh_approaches_per_target', 3
        )
        self.declare_parameter('stuck.window_s', 6.0)
        self.declare_parameter(
            'stuck.progress_threshold_m',
            0.05,
        )
        self.declare_parameter(
            'stuck.alignment_threshold_rad',
            0.39269908169872414,
        )
        self.declare_parameter('blacklist.radius_m', 0.75)
        self.declare_parameter('blacklist.duration_s', 30.0)
        self.declare_parameter(
            'blacklist.permanent_after_failures',
            2,
        )
        self.declare_parameter('visited.radius_m', 0.60)
        self.declare_parameter(
            'permanent_exclusion_radius_m',
            0.20,
        )
        self.declare_parameter(
            'selection.distance_slack_m',
            2.0,
        )
        self.declare_parameter(
            'planning.wall_closing_radius_m',
            0.05,
        )
        self.declare_parameter(
            'planning.unknown_clearance_m',
            0.10,
        )
        self.declare_parameter(
            'completion_debounce.period_s',
            8.0,
        )
        self.declare_parameter(
            'selection.approach_search_radius_m',
            1.5,
        )

        self.goal_reached_distance_m = (
            self.get_parameter('goal_reached_distance_m').value
        )
        self.maximum_goal_path_failures = (
            self.get_parameter(
                'maximum_goal_path_failures'
            ).value
        )
        self.maximum_fresh_approaches_per_target = (
            self.get_parameter(
                'maximum_fresh_approaches_per_target'
            ).value
        )
        self.stuck_window_s = (
            self.get_parameter('stuck.window_s').value
        )
        self.stuck_progress_threshold_m = (
            self.get_parameter(
                'stuck.progress_threshold_m'
            ).value
        )
        self.stuck_alignment_threshold_rad = (
            self.get_parameter(
                'stuck.alignment_threshold_rad'
            ).value
        )
        self.blacklist_radius_m = (
            self.get_parameter('blacklist.radius_m').value
        )
        self.blacklist_duration_s = (
            self.get_parameter('blacklist.duration_s').value
        )
        self.permanent_after_failures = (
            self.get_parameter(
                'blacklist.permanent_after_failures'
            ).value
        )
        self.visited_radius_m = (
            self.get_parameter('visited.radius_m').value
        )
        self.permanent_exclusion_radius_m = (
            self.get_parameter(
                'permanent_exclusion_radius_m'
            ).value
        )
        self.distance_slack_m = (
            self.get_parameter(
                'selection.distance_slack_m'
            ).value
        )
        self.wall_closing_radius_m = (
            self.get_parameter(
                'planning.wall_closing_radius_m'
            ).value
        )
        self.unknown_clearance_m = (
            self.get_parameter(
                'planning.unknown_clearance_m'
            ).value
        )
        self.completion_debounce_period_s = (
            self.get_parameter(
                'completion_debounce.period_s'
            ).value
        )
        self.approach_search_radius_m = (
            self.get_parameter(
                'selection.approach_search_radius_m'
            ).value
        )

        # Completion-debounce state: when no goal can be selected,
        # the planner waits one debounce period (letting SLAM settle)
        # before declaring exploration complete. This is a
        # deliberate planner stop -- never a stuck/recovery event,
        # and it publishes no commands.
        self.completion_debounce_active = False
        self.completion_debounce_started_s = 0.0
        self.exploration_complete_logged = False
        # True while completion is held open specifically because a
        # reachable frontier candidate is waiting out an active
        # temporary cooldown. Lets the deferral warning fire once on
        # entry rather than every cycle.
        self.completion_deferred_by_cooldown = False

        # Spatial goal memory (map-frame world coordinates):
        # temporary failed regions expire, permanent failed and
        # visited regions last for the whole node run.
        # Lifetime failure records: dicts with x, y,
        # failure_count, blocked_until_s. Counts survive cooldown
        # expiry so a later failure can still promote the region.
        self.failure_records = []
        self.permanent_failed_regions = []
        self.visited_goal_regions = []

        # Machine-readable mission counters. Increment exactly once
        # at the real lifecycle event (see update_goal_and_path,
        # stuck_check_callback, request_recovery) so the
        # /exploration_result JSON reflects true autonomy behavior.
        self.goals_assigned = 0
        self.goals_reached = 0
        self.failure_events = 0
        self.temporary_failure_events = 0
        self.recovery_requests = 0

        # Explicit completion state, published on /exploration_complete.
        self.exploration_complete = False

        # Progress samples for the committed goal only:
        # (time_s, x, y, yaw).
        self.progress_samples = deque()

        # The exact grid used for planning this map cycle; published
        # on /planning_grid so it can never silently diverge from
        # what BFS/A* consumed.
        self.last_planning_grid = None
        self.escape_corridor_cells = []

        # Per-cycle candidate funnel diagnostics, reported in the
        # status log: raw -> unique -> eligible -> reachable ->
        # selected.
        self.last_unique_candidate_count = 0
        self.last_visited_rejected_count = 0
        self.last_failed_rejected_count = 0
        self.last_duplicate_count = 0
        self.last_eligible_count = 0
        self.last_goal_distance_rejected_count = 0
        self.last_unreachable_cluster_count = 0
        self.last_approach_cells = []
        self.last_selected_cluster_size = 0
        # Separate temporary vs permanent rejection accounting so a
        # candidate excluded only by an active temporary cooldown is
        # distinguished from one permanently blacklisted. A temporary
        # rejection means "pending retry", never "exploration done".
        self.last_temporary_rejected_count = 0
        self.last_permanent_rejected_count = 0
        self.last_retry_exhausted_count = 0

        # Terminal-outcome classification (truthful completion vs
        # blocked). Populated by decide_terminal_outcome before the
        # completion debounce fires; read by publish_mission_result and
        # the terminal log. Defaults describe an unknown terminal state
        # so a missing decision can never masquerade as success.
        self.terminal_outcome = None
        self.terminal_blocked_reason = None
        self.terminal_geometric_frontier_cells = 0
        self.terminal_geometric_frontier_clusters = 0
        self.terminal_reachable_candidate_clusters = 0
        self.terminal_post_exclusion_eligible = 0
        self.terminal_temporary_rejected = 0
        self.terminal_permanent_rejected = 0
        self.terminal_retry_exhausted = 0

        # Latest deployable pose (map -> base_footprint TF) used by
        # the stuck detector. Ground truth is never consumed here.
        self.latest_pose = None

        # Coordination with obstacle_guard: a pure state machine
        # tracks the request-pending / recovery-active cycle so
        # planning stays blocked across the whole lifecycle.
        self.recovery_cycle = RecoveryCoordinationState()

        self.frontier_cells = set()
        self.frontier_clusters = []
        self.robot_grid_cell = None
        self.selected_frontier_cell = None
        self.current_grid_path = None
        self.committed_goal_world = None
        self.goal_path_failure_count = 0
        # Retry state belongs to one selected *frontier target*, not to its
        # current free-space approach. It is cleared only when that target's
        # lifecycle ends or a genuinely different target is selected.
        self.active_frontier_target_anchor = None
        self.attempted_approach_cells = set()
        self.attempted_approach_paths = set()
        self.fresh_approach_count = 0

        self.rover_length_m = 0.45
        self.rover_width_m = 0.30
        self.path_clearance_m = 0.05

        self.frontier_publisher = self.create_publisher(
            Marker,
            '/frontier_markers',
            10,
        )

        self.planning_grid_publisher = (
            self.create_publisher(
                OccupancyGrid,
                '/planning_grid',
                10,
            )
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

        # Deployable pose source for the stuck detector: the
        # map -> base_footprint TF lookup. Ground truth is not
        # consumed by this node at all.
        self.pose_timer = self.create_timer(
            0.5,
            self.pose_timer_callback,
        )

        self.recovery_request_publisher = (
            self.create_publisher(
                Bool,
                '/recovery_request',
                10,
            )
        )

        # Recovery status is stateful: transient-local durability
        # latches the latest status so a newly started or restarted
        # Frontier node immediately learns whether recovery is
        # running. Requests stay volatile because they are one-shot
        # events, not state.
        status_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.recovery_status_subscription = (
            self.create_subscription(
                Bool,
                '/recovery_status',
                self.recovery_status_callback,
                # Must match the guard's transient-local status
                # publisher so a newly started Frontier node receives
                # the latest latched state immediately.
                status_qos,
            )
        )

        # Explicit exploration completion state and structured result.
        # Both use transient-local + reliable + keep-last depth 1 so a
        # late subscriber (e.g. the mission evaluator or an RViz panel)
        # immediately receives the latched final state without waiting
        # for the next transition. These are state topics, not event
        # streams: they are published only on transition / completion.
        completion_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.exploration_complete_publisher = (
            self.create_publisher(
                Bool,
                '/exploration_complete',
                completion_qos,
            )
        )
        self.exploration_result_publisher = (
            self.create_publisher(
                String,
                '/exploration_result',
                completion_qos,
            )
        )

        # Publish the initial latched completion state (false) so the
        # topic is never empty before exploration finishes.
        initial_state = Bool()
        initial_state.data = False
        self.exploration_complete_publisher.publish(initial_state)

        self.stuck_timer = self.create_timer(
            1.0,
            self.stuck_check_callback,
        )

        self.path_publisher = self.create_publisher(
            Path,
            '/planned_path',
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

        robot_x = None
        robot_y = None

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

        rover_radius_m = math.hypot(
            self.rover_length_m / 2.0,
            self.rover_width_m / 2.0,
        )

        inflation_radius_cells = math.ceil(
            (rover_radius_m + self.path_clearance_m)
            / map_message.info.resolution
        )

        inflated_data = inflate_occupancy_grid(
            data=map_message.data,
            width=map_message.info.width,
            height=map_message.info.height,
            inflation_radius_cells=inflation_radius_cells,
        )

        # Conservatively condition the derived planning grid:
        # close one-cell wall pinholes, then pad unknown space with
        # its own small buffer. The raw /map data is never mutated.
        wall_closing_radius_cells = max(
            0,
            int(round(
                self.wall_closing_radius_m
                / map_message.info.resolution
            )),
        )
        unknown_padding_cells = max(
            0,
            int(round(
                self.unknown_clearance_m
                / map_message.info.resolution
            )),
        )

        conditioned_data = close_occupied_walls(
            map_message.data,
            map_message.info.width,
            map_message.info.height,
            closing_radius_cells=wall_closing_radius_cells,
        )
        conditioned_data = pad_unknown_space(
            conditioned_data,
            map_message.info.width,
            map_message.info.height,
            padding_radius_cells=unknown_padding_cells,
        )

        # Merge: a cell is planning-blocked when blocked by the
        # occupied inflation OR by the conservative conditioning.
        # Raw unknown cells are preserved as -1 (never traversable).
        planning_data = build_planning_grid(
            raw_data=map_message.data,
            inflated_data=inflated_data,
            conditioned_data=conditioned_data,
        )

        start_is_traversable = (
            self.robot_grid_cell is not None
            and is_traversable_grid_cell(
                data=planning_data,
                width=map_message.info.width,
                height=map_message.info.height,
                row=self.robot_grid_cell[0],
                column=self.robot_grid_cell[1],
            )
        )

        # Approach-cell selection happens inside
        # update_goal_and_path(), after the shared BFS tree exists:
        # each cluster searches for a reachable, planning-free
        # approach instead of an arbitrary free cell that may be
        # disconnected.

        self.update_goal_and_path(
            map_message=map_message,
            raw_data=map_message.data,
            planning_data=planning_data,
            robot_x=robot_x,
            robot_y=robot_y,
        )

        # Publish the exact grid the planner just consumed (already
        # escape-adjusted if a corridor was opened) so the RViz view
        # can never diverge from BFS/A* reality.
        self.last_planning_grid = list(planning_data)

        planning_grid_message = OccupancyGrid()
        planning_grid_message.header = map_message.header
        planning_grid_message.info = map_message.info
        planning_grid_message.data = [
            int(value) for value in planning_data
        ]

        self.planning_grid_publisher.publish(
            planning_grid_message
        )

        path_message = Path()
        path_message.header = map_message.header

        if self.current_grid_path is not None:
            for row, column in self.current_grid_path:
                world_x, world_y = grid_cell_center(
                    row=row,
                    column=column,
                    resolution=map_message.info.resolution,
                    origin_x=map_message.info.origin.position.x,
                    origin_y=map_message.info.origin.position.y,
                )

                path_pose = PoseStamped()
                path_pose.header = map_message.header

                path_pose.pose.position.x = world_x
                path_pose.pose.position.y = world_y
                path_pose.pose.position.z = 0.0
                path_pose.pose.orientation.w = 1.0

                path_message.poses.append(path_pose)

        self.path_publisher.publish(path_message)

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

        selected_marker = Marker()
        selected_marker.header = map_message.header
        selected_marker.ns = 'selected_frontier'
        selected_marker.id = 0

        if self.selected_frontier_cell is None:
            selected_marker.action = Marker.DELETE
        else:
            selected_row, selected_column = (
                self.selected_frontier_cell
            )

            selected_x, selected_y = grid_cell_center(
                row=selected_row,
                column=selected_column,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            selected_marker.type = Marker.SPHERE
            selected_marker.action = Marker.ADD

            selected_marker.pose.position.x = selected_x
            selected_marker.pose.position.y = selected_y
            selected_marker.pose.position.z = 0.15
            selected_marker.pose.orientation.w = 1.0

            selected_marker.scale.x = 0.25
            selected_marker.scale.y = 0.25
            selected_marker.scale.z = 0.25

            selected_marker.color.r = 0.0
            selected_marker.color.g = 0.0
            selected_marker.color.b = 1.0
            selected_marker.color.a = 1.0

        path_marker = Marker()
        path_marker.header = map_message.header
        path_marker.ns = 'planned_path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.pose.orientation.w = 1.0

        path_marker.scale.x = 0.04

        path_marker.color.r = 1.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0

        if self.current_grid_path is None:
            path_marker.action = Marker.DELETE
        else:
            path_marker.action = Marker.ADD

            for row, column in self.current_grid_path:
                world_x, world_y = grid_cell_center(
                    row=row,
                    column=column,
                    resolution=map_message.info.resolution,
                    origin_x=map_message.info.origin.position.x,
                    origin_y=map_message.info.origin.position.y,
                )

                path_point = Point()
                path_point.x = world_x
                path_point.y = world_y
                path_point.z = 0.20

                path_marker.points.append(path_point)

        # Candidate markers show the actual approach cells used by
        # the planner (one per reachable cluster approach).
        for candidate_row, candidate_column in (
            self.last_approach_cells
        ):
            candidate_x, candidate_y = grid_cell_center(
                row=candidate_row,
                column=candidate_column,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            candidate_point = Point()
            candidate_point.x = candidate_x
            candidate_point.y = candidate_y
            candidate_point.z = 0.10

            candidate_marker.points.append(candidate_point)

        for cluster in self.frontier_clusters:
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

        memory_markers = self.build_memory_markers(
            map_message=map_message
        )

        self.frontier_publisher.publish(marker)
        self.frontier_publisher.publish(candidate_marker)
        self.frontier_publisher.publish(selected_marker)
        self.frontier_publisher.publish(path_marker)

        for memory_marker in memory_markers:
            self.frontier_publisher.publish(memory_marker)

        path_cells = (
            0 if self.current_grid_path is None
            else len(self.current_grid_path)
        )

        self.get_logger().info(
            f'map={map_message.info.width}x{map_message.info.height} '
            f'frontier_cells={len(self.frontier_cells)} '
            f'frontier_clusters={len(self.frontier_clusters)} '
            f'robot_grid_cell={self.robot_grid_cell} '
            f'selected_frontier_cell={self.selected_frontier_cell} '
            f'path_cells={path_cells} '
            f'inflation_radius_cells={inflation_radius_cells} '
            f'committed_goal_world={self.committed_goal_world} '
            f'start_traversable={start_is_traversable} '
            f'approach_candidates={len(self.last_approach_cells)} '
            f'visited_rejected={self.last_visited_rejected_count} '
            f'failed_rejected={self.last_failed_rejected_count} '
            f'temporary_rejected={self.last_temporary_rejected_count} '
            f'permanent_rejected={self.last_permanent_rejected_count} '
            f'unreachable_clusters={self.last_unreachable_cluster_count} '
            f'duplicate_candidates={self.last_duplicate_count} '
            f'goal_distance_rejected={self.last_goal_distance_rejected_count} '
            f'eligible_candidates={self.last_eligible_count} '
            f'temp_failed={self.active_cooldown_count()} '
            f'permanent_failed={len(self.permanent_failed_regions)} '
            f'visited_regions={len(self.visited_goal_regions)} '
            f'selected_cluster_size={self.last_selected_cluster_size} '
            f'recovery_pending={self.recovery_cycle.request_pending} '
            f'recovery_active={self.recovery_cycle.recovery_active}'
        )

    def recovery_status_callback(self, status):
        if status.data:
            # Becoming active is a state transition, not a cycle
            # end; no completion log and no window reset here.
            self.recovery_cycle.on_status_active()
            return

        ended = self.recovery_cycle.on_status_inactive()

        if ended:
            # Recovery finished or aborted: rearm the request latch
            # so a future distinct stuck event can request again,
            # and give the next assigned goal a fresh progress
            # window. Fires exactly once per cycle.
            self.reset_goal_progress()

            self.get_logger().info(
                'Recovery cycle ended; '
                'stuck detection window reset'
            )

    def request_recovery(self):
        """Publish a one-shot recovery request to obstacle_guard."""
        should_publish = (
            self.recovery_cycle.publish_request()
        )

        if not should_publish:
            return

        # The pending state is set BEFORE publishing so goal
        # assignment is blocked during the race before the guard's
        # active status arrives.
        request = Bool()
        request.data = True

        self.recovery_request_publisher.publish(request)
        self.recovery_requests += 1

    def pose_timer_callback(self):
        try:
            robot_transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                Time(),
            )
        except TransformException:
            return

        translation = robot_transform.transform.translation
        rotation = robot_transform.transform.rotation

        self.latest_pose = (
            translation.x,
            translation.y,
            quaternion_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
        )

    def node_time_s(self):
        return self.get_clock().now().nanoseconds / 1_000_000_000

    def set_exploration_complete(self, value):
        """
        Centralized completion-state transition.

        Publishes /exploration_complete only on a change, and emits the
        one-shot /exploration_result JSON when transitioning to True.
        Never issues velocity commands or touches the planner.
        """
        if bool(value) == self.exploration_complete:
            return

        self.exploration_complete = bool(value)

        state = Bool()
        state.data = self.exploration_complete
        self.exploration_complete_publisher.publish(state)

        if self.exploration_complete:
            self.clear_frontier_target_retry_state()
            self.publish_mission_result(
                completion_time_s=self.node_time_s()
            )

    def publish_mission_result(self, completion_time_s):
        """
        Emit the deterministic mission-result JSON once at completion.

        The ``completed`` flag stays True whenever the rover reaches a
        terminal stop (it reflects that the run ended, not that every
        frontier was resolved). ``outcome`` disambiguates genuine success
        from a blocked terminal state: 'success' means no geometric
        frontier remained; 'blocked' means substantial frontier remained
        but bounded recovery was exhausted (no valid approach survived).
        Existing readers that only check ``completed`` keep working; new
        readers should consult ``outcome``.
        """
        result = {
            'schema_version': 1,
            'completed': True,
            'outcome': self.terminal_outcome or 'success',
            'blocked_reason': self.terminal_blocked_reason,
            'completion_time_s': completion_time_s,
            'goals_assigned': self.goals_assigned,
            'goals_reached': self.goals_reached,
            'failure_events': self.failure_events,
            'temporary_failure_events': self.temporary_failure_events,
            'permanent_failed_regions': (
                len(self.permanent_failed_regions)
            ),
            'recovery_requests': self.recovery_requests,
            'visited_regions': len(self.visited_goal_regions),
            'frontier_cells': len(self.frontier_cells),
            'frontier_clusters': len(self.frontier_clusters),
            'geometric_frontier_cells': (
                self.terminal_geometric_frontier_cells
            ),
            'geometric_frontier_clusters': (
                self.terminal_geometric_frontier_clusters
            ),
            'reachable_candidate_clusters': (
                self.terminal_reachable_candidate_clusters
            ),
            'post_exclusion_eligible': (
                self.terminal_post_exclusion_eligible
            ),
        }

        message = String()
        message.data = json.dumps(result, sort_keys=True)
        self.exploration_result_publisher.publish(message)

        self.get_logger().info(
            f'Exploration result: {message.data}'
        )

    def reset_candidate_funnel(self):
        """Zero all per-cycle candidate funnel diagnostics."""
        self.last_unique_candidate_count = 0
        self.last_visited_rejected_count = 0
        self.last_failed_rejected_count = 0
        self.last_duplicate_count = 0
        self.last_goal_distance_rejected_count = 0
        self.last_unreachable_cluster_count = 0
        self.last_eligible_count = 0
        self.last_approach_cells = []
        self.last_selected_cluster_size = 0
        self.last_temporary_rejected_count = 0
        self.last_permanent_rejected_count = 0
        self.last_retry_exhausted_count = 0

    def reset_goal_progress(self):
        """Clear the progress window for the committed goal."""
        self.progress_samples.clear()

    def clear_frontier_target_retry_state(self):
        """Forget retry evidence after this target's lifecycle ends."""
        self.active_frontier_target_anchor = None
        self.attempted_approach_cells = set()
        self.attempted_approach_paths = set()
        self.fresh_approach_count = 0
        self.goal_path_failure_count = 0

    def start_frontier_target_attempt(self, anchor, approach_cell, path):
        """Start retry accounting for a genuinely newly selected target."""
        self.active_frontier_target_anchor = anchor
        self.attempted_approach_cells = {approach_cell}
        self.attempted_approach_paths = {tuple(path)}
        self.fresh_approach_count = 0
        self.goal_path_failure_count = 0

    def fresh_approach_for_failed_goal(
        self,
        map_message,
        raw_data,
        planning_data,
        bfs,
        width,
        height,
    ):
        """
        Retry a fresh collision-checked plan to the same cluster.

        A committed goal can fail its reconstruction either because the
        destination is genuinely unreachable or because the path was
        computed against a map that has since changed. Before treating
        the cluster as exhausted, look for an unattempted reachable approach
        for the exact selected target anchor. An approach or reconstructed
        path already tried for this target is never accepted again.
        """
        if bfs is None or not self.frontier_clusters:
            return None

        target_anchor = self.active_frontier_target_anchor
        if target_anchor is None:
            return None
        if (
            self.fresh_approach_count
            >= self.maximum_fresh_approaches_per_target
        ):
            return None

        max_approach_radius = max(
            1,
            int(round(
                self.approach_search_radius_m
                / map_message.info.resolution
            )),
        )

        target_cluster = next(
            (cluster for cluster in self.frontier_clusters
             if target_anchor in cluster),
            None,
        )
        if target_cluster is None:
            return None

        excluded_cells = set(self.attempted_approach_cells)
        while len(excluded_cells) < width * height:
            approach_cell = find_cluster_approach_cell_reachable(
                raw_data=raw_data,
                planning_data=planning_data,
                width=width,
                height=height,
                cluster=target_cluster,
                bfs=bfs,
                max_search_radius_cells=max_approach_radius,
                excluded_approach_cells=excluded_cells,
            )

            if approach_cell is None:
                return None

            path = reconstruct_grid_path(
                bfs['came_from'], approach_cell
            )
            if path is not None and tuple(path) not in (
                self.attempted_approach_paths
            ):
                return approach_cell, path
            excluded_cells.add(approach_cell)

        return None

    def abandon_path_invalid_target(self, goal_x, goal_y, now_s):
        """Register an exhausted target through the scoped failure path."""
        outcome = record_failure(
            failure_records=self.failure_records,
            permanent_regions=self.permanent_failed_regions,
            x=goal_x,
            y=goal_y,
            now_s=now_s,
            match_radius_m=self.blacklist_radius_m,
            blacklist_duration_s=self.blacklist_duration_s,
            promotion_failures=self.permanent_after_failures,
            promote_radius_m=self.permanent_exclusion_radius_m,
        )
        self.log_failure(outcome, goal_x, goal_y)
        self.failure_events += 1
        if outcome != 'promoted':
            self.temporary_failure_events += 1
        self.committed_goal_world = None
        # A temporary scoped cooldown is not permanent abandonment of this
        # target. Keep its retry budget so a later eligible approach cannot
        # restart the fresh-approach cap; promotion ends the target attempt.
        if outcome == 'promoted':
            self.clear_frontier_target_retry_state()
        self.reset_goal_progress()

    def stuck_check_callback(self):
        # While a recovery request is pending or a maneuver runs,
        # the planner must not assign or monitor a goal and no
        # samples accumulate.
        if self.recovery_cycle.planning_blocked:
            return

        # Samples are only collected while a goal is committed, so
        # no stale window accumulates between goals.
        if (
            self.committed_goal_world is None
            or self.latest_pose is None
        ):
            return

        now_s = self.node_time_s()
        position_x, position_y, yaw = self.latest_pose

        self.progress_samples.append(
            (now_s, position_x, position_y, yaw)
        )

        minimum_sample_time_s = now_s - self.stuck_window_s

        while (
            self.progress_samples
            and self.progress_samples[0][0]
            < minimum_sample_time_s
        ):
            self.progress_samples.popleft()

        stuck = is_stuck(
            progress_samples=self.progress_samples,
            goal_position=self.committed_goal_world,
            minimum_window_s=(
                self.stuck_window_s - 1.5
            ),
            progress_threshold_m=(
                self.stuck_progress_threshold_m
            ),
            alignment_threshold_rad=(
                self.stuck_alignment_threshold_rad
            ),
        )

        if not stuck:
            return

        goal_x, goal_y = self.committed_goal_world

        # Register this stuck failure in spatial memory. A repeat
        # failure near the same region promotes it to permanent.
        outcome = record_failure(
            failure_records=self.failure_records,
            permanent_regions=self.permanent_failed_regions,
            x=goal_x,
            y=goal_y,
            now_s=now_s,
            match_radius_m=self.blacklist_radius_m,
            blacklist_duration_s=self.blacklist_duration_s,
            promotion_failures=self.permanent_after_failures,
            promote_radius_m=self.permanent_exclusion_radius_m,
        )
        self.log_failure(outcome, goal_x, goal_y)

        # One failure registered at this exact lifecycle point.
        self.failure_events += 1
        if outcome != 'promoted':
            self.temporary_failure_events += 1

        self.committed_goal_world = None
        # A temporary stuck cooldown leaves this target's bounded retry
        # state intact. Only permanent promotion abandons the target.
        if outcome == 'promoted':
            self.clear_frontier_target_retry_state()
        self.reset_goal_progress()

        # Ask obstacle_guard to run an escape maneuver for this
        # generic stuck event (side wedge, traction loss, ...). The
        # guard remains the only publisher of final commands and
        # still enforces every sensor safety check.
        self.request_recovery()

        self.get_logger().warning(
            f'Rover made no progress toward goal '
            f'({goal_x:.12f}, {goal_y:.12f}); '
            f'recovery requested'
        )

    def update_goal_and_path(
        self,
        map_message,
        raw_data,
        planning_data,
        robot_x,
        robot_y,
    ):
        self.selected_frontier_cell = None
        self.current_grid_path = None

        # Reset funnel diagnostics before any early return so no
        # status line can ever report stale counts.
        self.reset_candidate_funnel()
        self.escape_corridor_cells = []

        # No new goal may be planned while a recovery request is
        # pending or a maneuver is active; assignment waits for the
        # whole recovery cycle to end.
        if self.recovery_cycle.planning_blocked:
            return

        if (
            self.robot_grid_cell is None
            or robot_x is None
            or robot_y is None
        ):
            return

        width = map_message.info.width
        height = map_message.info.height

        plan_start = self.robot_grid_cell

        if not is_traversable_grid_cell(
            data=planning_data,
            width=width,
            height=height,
            row=plan_start[0],
            column=plan_start[1],
        ):
            # The rover sits inside a derived-blocked zone (wall
            # inflation, pinhole closing, or unknown padding). Find
            # an escape corridor that walks only through raw-free
            # cells from the rover's real cell to planning-safe
            # space.
            escape_path = find_escape_path(
                raw_data=raw_data,
                inflated_data=planning_data,
                width=width,
                height=height,
                start=plan_start,
            )

            if escape_path is None:
                return

            # Temporarily clear only the raw-free cells of the
            # escape corridor so A* can plan through them from the
            # real rover cell. Every cleared cell is free on the
            # raw map, so the published path stays physically
            # connected to the rover. This adjusted grid IS the
            # grid stored and published below — planning, memory,
            # and RViz can never diverge.
            for row, column in escape_path[:-1]:
                index = row * width + column
                planning_data[index] = 0

            self.escape_corridor_cells = list(
                escape_path[:-1]
            )

        # One authoritative BFS per cycle: reachability, route
        # costs, cluster approach search, goal selection, and path
        # reconstruction all share this single tree. Failing to
        # compute it fails CLOSED: planning stops rather than
        # optimistically treating every candidate as reachable.
        bfs = compute_reachable_component(
            data=planning_data,
            width=width,
            height=height,
            start=plan_start,
        )

        if bfs is None:
            self.get_logger().warning(
                'Rover start cell is not traversable on the '
                'planning grid; refusing to select a goal '
                '(fail-closed)'
            )
            return

        # Per-cluster reachable approach selection: each cluster
        # searches for a planning-free approach cell inside the
        # rover's component, bounded so it cannot map onto an
        # unrelated cell across the map.
        max_approach_radius = max(
            1,
            int(round(
                self.approach_search_radius_m
                / map_message.info.resolution
            )),
        )

        candidate_cluster_sizes = {}
        candidate_target_anchors = {}
        duplicate_candidate_count = 0
        unreachable_cluster_count = 0

        for cluster in self.frontier_clusters:
            # Keep the original anchor whenever the map's updated cluster
            # still contains it. Cluster growth/reordering must not create a
            # new target identity or reset its retry budget.
            target_anchor = (
                self.active_frontier_target_anchor
                if self.active_frontier_target_anchor in cluster
                else (min(cluster) if cluster else None)
            )
            excluded_approaches = (
                self.attempted_approach_cells
                if target_anchor == self.active_frontier_target_anchor
                else None
            )
            approach_cell = (
                find_cluster_approach_cell_reachable(
                    raw_data=raw_data,
                    planning_data=planning_data,
                    width=width,
                    height=height,
                    cluster=cluster,
                    bfs=bfs,
                    max_search_radius_cells=(
                        max_approach_radius
                    ),
                    excluded_approach_cells=excluded_approaches,
                )
            )

            if approach_cell is None:
                # The whole cluster is disconnected from the rover's
                # component (or has no safe standoff within bound).
                unreachable_cluster_count += 1
                continue

            if approach_cell in candidate_cluster_sizes:
                duplicate_candidate_count += 1

            previous_size = candidate_cluster_sizes.get(
                approach_cell, 0
            )
            if len(cluster) > previous_size:
                candidate_cluster_sizes[approach_cell] = len(cluster)
                # The anchor is deliberately a frontier cell, separate from
                # the free-space approach, so recovery cannot drift to a
                # merely nearby cluster.
                candidate_target_anchors[approach_cell] = target_anchor

        unique_candidates = list(
            candidate_cluster_sizes.keys()
        )

        now_s = self.node_time_s()

        # Expire old cooldowns; lifetime failure counts survive so
        # a later failure near the same region can still promote.
        prune_expired_cooldowns(
            failure_records=self.failure_records,
            now_s=now_s,
        )

        if self.committed_goal_world is not None:
            goal_world_x, goal_world_y = (
                self.committed_goal_world
            )

            goal_distance = math.hypot(
                goal_world_x - robot_x,
                goal_world_y - robot_y,
            )

            if goal_distance <= self.goal_reached_distance_m:
                # Success: remember this region as visited and give
                # the next goal a fresh window. Never a failure.
                self.visited_goal_regions.append(
                    (goal_world_x, goal_world_y)
                )
                self.goals_reached += 1
                self.get_logger().info(
                    f'Goal reached at '
                    f'({goal_world_x:.2f}, {goal_world_y:.2f}); '
                    f'region marked visited'
                )
                self.committed_goal_world = None
                self.clear_frontier_target_retry_state()
                self.reset_goal_progress()
            else:
                goal_row, goal_column = world_point_to_grid_cell(
                    world_x=goal_world_x,
                    world_y=goal_world_y,
                    resolution=map_message.info.resolution,
                    origin_x=map_message.info.origin.position.x,
                    origin_y=map_message.info.origin.position.y,
                )

                # Reconstruct the committed path from the shared BFS
                # tree: no second search is performed. A committed
                # world goal can convert outside the current resized
                # map; test membership in the cost map instead of an
                # index into the reachable grid so an out-of-range index
                # can never crash or read the wrong cell.
                goal_cell = (goal_row, goal_column)

                committed_path = None

                if goal_cell in bfs['cost']:
                    committed_path = reconstruct_grid_path(
                        bfs['came_from'],
                        goal_cell,
                    )

                if committed_path is not None:
                    self.selected_frontier_cell = (
                        goal_row,
                        goal_column,
                    )
                    self.current_grid_path = committed_path
                    return

                self.goal_path_failure_count += 1

                self.get_logger().warning(
                    f'Committed path invalid '
                    f'({self.goal_path_failure_count}/'
                    f'{self.maximum_goal_path_failures}) for goal '
                    f'({goal_world_x:.2f}, {goal_world_y:.2f})'
                )

                if (
                    self.goal_path_failure_count
                    < self.maximum_goal_path_failures
                ):
                    return

                # Bounded recovery: retry the exact SAME target anchor.
                # Every accepted approach/path remains excluded and this
                # target-wide failure count is deliberately not reset. The
                # finite grid therefore yields only finitely many retries;
                # exhausted alternatives fall through to scoped failure
                # registration and the normal terminal-decision flow.
                fresh = self.fresh_approach_for_failed_goal(
                    map_message=map_message,
                    raw_data=raw_data,
                    planning_data=planning_data,
                    bfs=bfs,
                    width=width,
                    height=height,
                )
                if fresh is not None:
                    approach_cell, path = fresh
                    self.selected_frontier_cell = approach_cell
                    self.current_grid_path = path
                    self.committed_goal_world = (
                        grid_cell_center(
                            row=approach_cell[0],
                            column=approach_cell[1],
                            resolution=map_message.info.resolution,
                            origin_x=map_message.info.origin.position.x,
                            origin_y=map_message.info.origin.position.y,
                        )
                    )
                    self.goals_assigned += 1
                    self.attempted_approach_cells.add(approach_cell)
                    self.attempted_approach_paths.add(tuple(path))
                    self.fresh_approach_count += 1
                    self.reset_goal_progress()
                    self.completion_debounce_active = False
                    if self.exploration_complete:
                        self.set_exploration_complete(False)
                    self.get_logger().info(
                        f'Fresh approach committed for failed goal '
                        f'({goal_world_x:.2f}, {goal_world_y:.2f}); '
                        f'retrying same cluster via new route'
                    )
                    return

                # Path-invalid abandonment: register a failed attempt but
                # do NOT request physical recovery; the stuck detector owns
                # recovery requests. Scope the permanent exclusion to the
                # actual failed approach (0.20 m) so it cannot blanket the
                # whole frontier cluster and eliminate its other approaches.
                self.abandon_path_invalid_target(
                    goal_world_x, goal_world_y, now_s
                )

        # Candidates are already unique and reachable: approach
        # selection ran against the shared BFS component and
        # deduplicated cluster collisions.
        eligible_candidates = []
        visited_rejected_count = 0
        failed_rejected_count = 0
        temporary_rejected_count = 0
        permanent_rejected_count = 0
        retry_exhausted_count = 0
        goal_distance_rejected_count = 0

        unique_candidates = list(
            candidate_cluster_sizes.keys()
        )

        for candidate_row, candidate_column in (
            unique_candidates
        ):
            candidate_x, candidate_y = grid_cell_center(
                row=candidate_row,
                column=candidate_column,
                resolution=map_message.info.resolution,
                origin_x=map_message.info.origin.position.x,
                origin_y=map_message.info.origin.position.y,
            )

            exclusion = is_excluded(
                x=candidate_x,
                y=candidate_y,
                failure_records=self.failure_records,
                permanent_regions=(
                    self.permanent_failed_regions
                ),
                visited_regions=self.visited_goal_regions,
                now_s=now_s,
                exclusion_radius_m=self.blacklist_radius_m,
                temporary_radius_m=self.permanent_exclusion_radius_m,
                visited_radius_m=self.visited_radius_m,
            )

            if exclusion == 'permanent':
                failed_rejected_count += 1
                permanent_rejected_count += 1
                continue

            if exclusion == 'temporary':
                failed_rejected_count += 1
                temporary_rejected_count += 1
                continue

            if exclusion == 'visited':
                visited_rejected_count += 1
                self.get_logger().debug(
                    f'Candidate ({candidate_x:.2f}, '
                    f'{candidate_y:.2f}) rejected: already visited'
                )
                continue

            if (
                candidate_target_anchors[candidate_row, candidate_column]
                == self.active_frontier_target_anchor
                and self.fresh_approach_count
                >= self.maximum_fresh_approaches_per_target
            ):
                # Cap exhaustion is a truthful, target-scoped rejection. Do
                # not let it fall through to the misleading visited/too-close
                # terminal reason; unrelated targets remain eligible below.
                retry_exhausted_count += 1
                continue

            candidate_distance = math.hypot(
                candidate_x - robot_x,
                candidate_y - robot_y,
            )

            if candidate_distance <= self.goal_reached_distance_m:
                goal_distance_rejected_count += 1
                continue

            # Already guaranteed reachable: the approach search only
            # returns cells inside the BFS component.
            eligible_candidates.append(
                (candidate_row, candidate_column)
            )

        # Persist the funnel for the status log emitted by
        # map_callback after this method returns.
        self.last_unique_candidate_count = len(
            unique_candidates
        )
        self.last_visited_rejected_count = (
            visited_rejected_count
        )
        self.last_failed_rejected_count = (
            failed_rejected_count
        )
        self.last_temporary_rejected_count = (
            temporary_rejected_count
        )
        self.last_permanent_rejected_count = (
            permanent_rejected_count
        )
        self.last_retry_exhausted_count = retry_exhausted_count
        self.last_duplicate_count = duplicate_candidate_count
        self.last_goal_distance_rejected_count = (
            goal_distance_rejected_count
        )
        self.last_eligible_count = len(eligible_candidates)
        self.last_approach_cells = list(unique_candidates)
        self.last_unreachable_cluster_count = (
            unreachable_cluster_count
        )

        distance_slack_cells = max(
            1,
            int(round(
                self.distance_slack_m
                / map_message.info.resolution
            )),
        )

        selection_result = select_cluster_weighted_goal(
            bfs=bfs,
            candidate_costs={
                candidate: (
                    candidate_cluster_sizes.get(candidate, 1),
                )
                for candidate in eligible_candidates
            },
            distance_slack_cells=distance_slack_cells,
        )

        if selection_result is None:
            # No cluster produced a selectable goal this cycle. Whether
            # this means exploration is genuinely finished depends on
            # WHY nothing was selectable. A candidate rejected only by
            # an active temporary cooldown is merely pending retry, not
            # permanently done: it must never count as completion.
            if self.last_temporary_rejected_count > 0:
                # Defer completion: a reachable frontier candidate is
                # still waiting for its temporary cooldown to expire.
                # Cancel any in-flight debounce and keep
                # /exploration_complete false. Publish no command.
                self.completion_debounce_active = False
                self.completion_debounce_started_s = 0.0
                self.exploration_complete_logged = False
                if self.exploration_complete:
                    self.set_exploration_complete(False)
                if not self.completion_deferred_by_cooldown:
                    self.completion_deferred_by_cooldown = True
                    self.get_logger().warning(
                        'Completion deferred: reachable frontier '
                        f'candidate(s) rejected by active temporary '
                        f'cooldown '
                        f'(temp_rejected='
                        f'{self.last_temporary_rejected_count}); '
                        'awaiting cooldown expiry, planner holding, '
                        'no commands issued'
                    )
                return

            # No temporary-blocking candidate and none selectable: decide
            # the terminal outcome truthfully. Distinguish genuinely
            # finished exploration (no geometric frontier remains) from
            # stopped-with-unresolved-frontiers (frontier remains but every
            # candidate is permanently/temporarily blacklisted and bounded
            # recovery is exhausted). These are three distinct facts and
            # must not all be called "selectable".
            self.completion_deferred_by_cooldown = False
            self.decide_terminal_outcome(
                geometric_frontier_cells=len(self.frontier_cells),
                geometric_frontier_clusters=len(
                    self.frontier_clusters
                ),
                reachable_candidate_clusters=len(
                    unique_candidates
                ),
                post_exclusion_eligible=self.last_eligible_count,
                temporary_rejected=self.last_temporary_rejected_count,
                permanent_rejected=self.last_permanent_rejected_count,
                retry_exhausted=self.last_retry_exhausted_count,
            )
            self.completion_debounce_tick()
            return

        (
            self.selected_frontier_cell,
            self.current_grid_path,
        ) = selection_result

        selected_row, selected_column = (
            self.selected_frontier_cell
        )

        selected_x, selected_y = grid_cell_center(
            row=selected_row,
            column=selected_column,
            resolution=map_message.info.resolution,
            origin_x=map_message.info.origin.position.x,
            origin_y=map_message.info.origin.position.y,
        )

        self.committed_goal_world = (
            selected_x,
            selected_y,
        )
        self.goals_assigned += 1

        # A valid goal was assigned: exploration is demonstrably not
        # complete. If we had previously declared completion (e.g. a
        # new selectable frontier appeared after a map update),
        # transition the state back to false.
        self.completion_deferred_by_cooldown = False
        if self.exploration_complete:
            self.set_exploration_complete(False)
        selected_anchor = candidate_target_anchors[
            self.selected_frontier_cell
        ]
        if selected_anchor == self.active_frontier_target_anchor:
            # This is a post-cooldown approach to the same target. Preserve
            # its cap and append the new approach instead of restarting it.
            self.attempted_approach_cells.add(self.selected_frontier_cell)
            self.attempted_approach_paths.add(
                tuple(self.current_grid_path)
            )
            self.fresh_approach_count += 1
        else:
            self.start_frontier_target_attempt(
                anchor=selected_anchor,
                approach_cell=self.selected_frontier_cell,
                path=self.current_grid_path,
            )
        self.reset_goal_progress()
        self.last_selected_cluster_size = (
            candidate_cluster_sizes.get(
                self.selected_frontier_cell, 1
            )
        )

        # A goal was assigned: the debounce is over and exploration
        # is demonstrably not complete.
        self.completion_debounce_active = False
        self.exploration_complete_logged = False

        self.get_logger().info(
            f'Goal assigned at ({selected_x:.2f}, '
            f'{selected_y:.2f}) cluster_size='
            f'{candidate_cluster_sizes.get(self.selected_frontier_cell, 1)}'
        )

    def active_cooldown_count(self):
        """Count failure records with an active cooldown."""
        now_s = self.node_time_s()

        return sum(
            1
            for record in self.failure_records
            if (
                record['blocked_until_s']
                != float('inf')
                and now_s < record['blocked_until_s']
            )
        )

    def build_memory_markers(self, map_message):
        """Sphere markers for failure/visited memory regions."""
        # Diameters correspond to each region's exclusion radius:
        # temporary failed = orange (0.75), permanent failed =
        # magenta (0.75), visited = muted gray-green (0.60).

        now_s = self.node_time_s()

        # Temporary regions are derived live from failure records:
        # any record with an active (finite, unexpired) cooldown.
        # Promoted records carry inf and are excluded here because
        # they already appear in the permanent marker list.
        temporary_regions = [
            (record['x'], record['y'])
            for record in self.failure_records
            if (
                record['blocked_until_s']
                != float('inf')
                and now_s < record['blocked_until_s']
            )
        ]

        markers = []

        definitions = [
            (
                'temp_failed',
                temporary_regions,
                1.0, 0.55, 0.0,
                self.blacklist_radius_m * 2.0,
            ),
            (
                'permanent_failed',
                self.permanent_failed_regions,
                0.8, 0.0, 0.8,
                self.blacklist_radius_m * 2.0,
            ),
            (
                'visited',
                self.visited_goal_regions,
                0.45, 0.55, 0.45,
                self.visited_radius_m * 2.0,
            ),
        ]

        for namespace, regions, red, green, blue, diameter in definitions:
            marker = Marker()
            marker.header = map_message.header
            marker.ns = namespace
            marker.id = 0
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0

            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = 0.05

            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 0.35

            for region in regions:
                point = Point()
                point.x = region[0]
                point.y = region[1]
                point.z = 0.02

                marker.points.append(point)

            markers.append(marker)

        return markers

    def decide_terminal_outcome(
        self,
        geometric_frontier_cells,
        geometric_frontier_clusters,
        reachable_candidate_clusters,
        post_exclusion_eligible,
        temporary_rejected,
        permanent_rejected,
        retry_exhausted=0,
    ):
        """
        Classify the terminal state truthfully from the three facts.

        * genuine success: no MEANINGFUL geometric frontier remains, i.e.
          no residual component contains >= 5 cells (small stray cells are
          within contract and do not block completion).
        * blocked: a meaningful (>= 5 cell) frontier component remains, yet
          no candidate is selectable. The reason refines the report:
            - every candidate is permanently/temporarily blacklisted
              (bounded recovery exhausted); or
            - the only rejects are visited / too-close cells, which are
              stable, non-transient rejections of an unresolved frontier
              and must NOT be dismissed as success.
          A pending temporary cooldown is handled upstream (the run is
          deferred, not terminated), so by the time this runs
          temporary_rejected is already 0.

        Stores the decision plus the raw counts so the result message and
        the terminal log report facts, never a blanket "selectable" claim.
        """
        self.terminal_geometric_frontier_cells = (
            geometric_frontier_cells
        )
        self.terminal_geometric_frontier_clusters = (
            geometric_frontier_clusters
        )
        self.terminal_reachable_candidate_clusters = (
            reachable_candidate_clusters
        )
        self.terminal_post_exclusion_eligible = (
            post_exclusion_eligible
        )
        self.terminal_temporary_rejected = temporary_rejected
        self.terminal_permanent_rejected = permanent_rejected
        self.terminal_retry_exhausted = retry_exhausted

        # Genuine success: no residual >= 5-cell component at all.
        if geometric_frontier_clusters == 0:
            self.terminal_outcome = 'success'
            self.terminal_blocked_reason = None
            return

        # A meaningful (>= 5 cell) frontier component remains and nothing is
        # eligible. This is stopped-with-unresolved-frontiers, never success.
        # Distinguish the rejection cause only for the human-readable reason.
        if retry_exhausted > 0:
            reason = (
                'geometric frontier remains (>= 5 cells in a '
                'component) but the active target exhausted its '
                'fresh-approach retry cap'
            )
        elif permanent_rejected > 0:
            reason = (
                'geometric frontier remains (>= 5 cells in a '
                'component) but all candidates are permanently '
                'blacklisted; bounded recovery exhausted'
            )
        else:
            reason = (
                'geometric frontier remains (>= 5 cells in a '
                'component) but no candidate is eligible; remaining '
                'rejects are visited / too-close cells, which are '
                'stable and do not resolve on retry'
            )
        self.terminal_outcome = 'blocked'
        self.terminal_blocked_reason = reason

    def completion_debounce_tick(self):
        """Advance the completion debounce state machine."""
        now_s = self.node_time_s()

        if not self.completion_debounce_active:
            self.completion_debounce_active = True
            self.completion_debounce_started_s = now_s

            self.get_logger().warning(
                'No selectable goal; debouncing completion for '
                f'{self.completion_debounce_period_s:.0f}s '
                '(planner holding, no commands issued)'
            )
            return

        if (
            now_s - self.completion_debounce_started_s
            < self.completion_debounce_period_s
        ):
            return

        if not self.exploration_complete_logged:
            self.exploration_complete_logged = True

            # Report FACTUAL counts, never a claim that geometric
            # frontier is zero. Three distinct facts are reported
            # separately so a reader can tell genuine completion from
            # stopped-with-unresolved-frontiers.
            self.get_logger().warning(
                'Exploration complete (terminal decision after debounce): '
                f'outcome={self.terminal_outcome} '
                f'(geometric_frontier_cells='
                f'{self.terminal_geometric_frontier_cells}, '
                f'geometric_frontier_clusters='
                f'{self.terminal_geometric_frontier_clusters}, '
                f'reachable_candidate_clusters='
                f'{self.terminal_reachable_candidate_clusters}, '
                f'post_exclusion_eligible='
                f'{self.terminal_post_exclusion_eligible}, '
                f'temporary_rejected='
                f'{self.terminal_temporary_rejected}, '
                f'permanent_rejected='
                f'{self.terminal_permanent_rejected}, '
                f'visited={len(self.visited_goal_regions)}, '
                f'permanent_failed='
                f'{len(self.permanent_failed_regions)})'
                + (
                    f'; blocked_reason={self.terminal_blocked_reason}'
                    if self.terminal_outcome == 'blocked'
                    else ''
                )
            )

        # Declare machine-readable completion only when exploration is
        # genuinely finished. Never declare completion merely because
        # recovery is active/pending or a committed goal still exists;
        # those states mean the rover is still working, not done.
        if (
            not self.recovery_cycle.recovery_active
            and not self.recovery_cycle.request_pending
            and self.committed_goal_world is None
        ):
            self.set_exploration_complete(True)

    def log_failure(self, outcome, goal_x, goal_y):
        """Emit lifecycle logging for a registered failure."""
        if outcome == 'promoted':
            self.get_logger().warning(
                f'Goal region ({goal_x:.2f}, {goal_y:.2f}) '
                f'promoted to PERMANENT blacklist after repeated '
                f'failures'
            )
        else:
            self.get_logger().warning(
                f'Temporary failure recorded near '
                f'({goal_x:.2f}, {goal_y:.2f}); expires in '
                f'{self.blacklist_duration_s:.0f}s'
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
