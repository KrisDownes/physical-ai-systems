from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    enable_motion = LaunchConfiguration('enable_motion')

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
        PythonLaunchDescriptionSource(simulation_file)
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
            simulation,
            slam,
            frontier_detector,
            obstacle_guard,
            path_follower,
            ekf_filter_node,
        ]
    )
