from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare('rover_description')

    enable_rviz = LaunchConfiguration('enable_rviz')

    xacro_file = PathJoinSubstitution(
        [
            package_share,
            'urdf',
            'rover.urdf.xacro',
        ]
    )

    rviz_config_file = PathJoinSubstitution(
        [
            package_share,
            'rviz',
            'display.rviz',
        ]
    )

    robot_description_content = Command(
        [
            FindExecutable(name='xacro'),
            ' ',
            xacro_file,
        ]
    )

    robot_description = {
        'robot_description': robot_description_content,
    }

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d',
            rviz_config_file,
        ],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(enable_rviz),
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/kd_bot/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/ground_truth/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/imu/data_raw@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/kd_world/model/kd_bot/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        remappings=[
            ('/world/kd_world/model/kd_bot/joint_state', '/joint_states'),
            ('/camera', '/camera/image_raw'),
            ('/camera_info', '/camera/camera_info'),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'enable_rviz',
                default_value='true',
                description='Start RViz visualization',
            ),
            robot_state_publisher_node,
            rviz_node,
            bridge_node,
        ]
    )
