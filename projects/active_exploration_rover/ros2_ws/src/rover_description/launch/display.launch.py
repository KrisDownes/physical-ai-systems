from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("rover_description")

    xacro_file = PathJoinSubstitution(
        [
            package_share,
            "urdf",
            "rover.urdf.xacro",
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [
            package_share,
            "rviz",
            "display.rviz",
        ]
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            xacro_file,
        ]
    )

    robot_description = {
        "robot_description": robot_description_content,
    }

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            rviz_config_file,
        ],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            rviz_node,
        ]
    )
