from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("rover_description")
    gazebo_package_share = FindPackageShare("ros_gz_sim")

    gazebo_file = PathJoinSubstitution(
        [
            gazebo_package_share,
            "launch",
            "gz_sim.launch.py"
        ]
    )

    display_file = PathJoinSubstitution(
        [
            package_share,
            "launch",
            "display.launch.py",
        ]
    )
    world_file = PathJoinSubstitution(
        [
            package_share,
            "worlds",
            "kd_world.sdf",
        ]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_file),
        launch_arguments={
            "gz_args": ["-r ", world_file],
        }.items(),
    )

    display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(display_file)
    )
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "/robot_description",
            "-name", "kd_bot",
            "-z", "0.02",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            gazebo,
            display,
            spawn_robot,
        ]
    )
