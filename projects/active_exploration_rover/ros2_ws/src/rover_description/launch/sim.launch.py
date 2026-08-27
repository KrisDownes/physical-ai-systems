from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare('rover_description')
    gazebo_package_share = FindPackageShare('ros_gz_sim')

    gazebo_file = PathJoinSubstitution(
        [
            gazebo_package_share,
            'launch',
            'gz_sim.launch.py'
        ]
    )

    display_file = PathJoinSubstitution(
        [
            package_share,
            'launch',
            'display.launch.py',
        ]
    )
    world_file = PathJoinSubstitution(
        [
            package_share,
            'worlds',
            'kd_world.sdf',
        ]
    )

    # Robot spawn pose. Defaults preserve V14 behavior (z=0.02 lift).
    # Yaw is radians. Scenarios are set explicitly per run; there is
    # no random spawn selection.
    spawn_x = LaunchConfiguration('spawn_x')
    spawn_y = LaunchConfiguration('spawn_y')
    spawn_z = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_file),
        launch_arguments={
            'gz_args': ['-r ', world_file],
        }.items(),
    )

    display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(display_file)
    )
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'kd_bot',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', spawn_z,
            '-Y', spawn_yaw,
        ],
        output='screen',
    )

    return LaunchDescription(
        [
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
            gazebo,
            display,
            spawn_robot,
        ]
    )
