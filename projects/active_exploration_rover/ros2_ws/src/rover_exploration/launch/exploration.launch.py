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

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulation_file)
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_file),
        launch_arguments={
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

    exploration_controller = Node(
        package='rover_control',
        executable='exploration_controller',
        output='screen',
        parameters=[
            {'use_sim_time': True},
        ],
        condition=IfCondition(enable_motion),
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
            exploration_controller,
        ]
    )
