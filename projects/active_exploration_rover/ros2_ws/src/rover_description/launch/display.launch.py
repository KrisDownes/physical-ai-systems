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
        parameters=[robot_description, {"use_sim_time": True}],
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
        parameters=[{"use_sim_time": True}]
    )

    bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="bridge",
        output="screen",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/kd_bot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/model/kd_bot/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/world/kd_world/model/kd_bot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model",
        ],
        remappings=[
            ("/model/kd_bot/tf", "/tf"),
            ("/world/kd_world/model/kd_bot/joint_state", "/joint_states")
        ],
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            rviz_node,
            bridge_node,
        ]
    )
