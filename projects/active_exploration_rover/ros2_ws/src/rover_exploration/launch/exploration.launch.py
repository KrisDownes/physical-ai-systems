from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    enable_motion = LaunchConfiguration('enable_motion')
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    simulation_file = PathJoinSubstitution(
        [
            FindPackageShare('rover_description'),
            'launch',
            'sim.launch.py',
        ]
    )

    slam_file = PathJoinSubstitution(
        [
            FindPackageShare('slam_toolbox'),
            'launch',
            'online_async_launch.py',
        ]
    )

    slam_config_file = PathJoinSubstitution(
        [
            FindPackageShare('rover_exploration'),
            'config',
            'slam_toolbox.yaml',
        ]
    )

    ekf_config_file = PathJoinSubstitution(
        [
            FindPackageShare('rover_exploration'),
            'config',
            'ekf.yaml',
        ]
    )

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulation_file),
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_z': spawn_z,
            'spawn_yaw': spawn_yaw,
        }.items(),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_file),
        launch_arguments={
            'slam_params_file': slam_config_file,
            'use_sim_time': 'true',
        }.items(),
    )

    frontier_detector = Node(
        package='rover_exploration',
        executable='frontier_detector',
        output='screen',
        parameters=[
            {'use_sim_time': True},
        ],
    )

    obstacle_guard = Node(
        package='rover_control',
        executable='obstacle_guard',
        output='screen',
        parameters=[
            {'use_sim_time': True},
        ],
        condition=IfCondition(enable_motion),
    )

    path_follower = Node(
        package='rover_control',
        executable='path_follower',
        name='path_follower',
        output='screen',
        parameters=[
            {'use_sim_time': True},
        ],
        condition=IfCondition(enable_motion),
    )

    # Robot-localization EKF: the single publisher of odom -> base_footprint.
    # Starts with the simulation/SLAM stack (not gated on enable_motion) so that
    # motion-disabled mapping tests also get a clean odom frame.
    ekf_filter_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            ekf_config_file,
            {'use_sim_time': True},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'enable_motion',
                default_value='false',
                description='Start autonomous rover motion',
            ),
            DeclareLaunchArgument(
                'spawn_x',
                default_value='0.0',
                description='Robot spawn X position (meters)',
            ),
            DeclareLaunchArgument(
                'spawn_y',
                default_value='0.0',
                description='Robot spawn Y position (meters)',
            ),
            DeclareLaunchArgument(
                'spawn_z',
                default_value='0.02',
                description='Robot spawn Z height (meters)',
            ),
            DeclareLaunchArgument(
                'spawn_yaw',
                default_value='0.0',
                description='Robot spawn yaw (radians)',
            ),
            simulation,
            slam,
            frontier_detector,
            obstacle_guard,
            path_follower,
            ekf_filter_node,
        ]
    )
