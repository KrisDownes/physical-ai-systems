# Rover Model

## Purpose
- Mission: indoor autonomous exploration

## Coordinate Convention
- x forward, y left, z up
- Drive configuration: two driven wheels and one passive rear caster

## Physical Dimensions
- Chassis length: 0.45m
- Chassis width: 0.30m
- Chassis height: 0.15m
- Ground Clearance: 0.05m
- Mass: 8 kg
- Wheel radius: 0.075m
- Wheel width: 0.040m
- LiDAR height above ground: 0.22m
- Camera height above ground: 0.20m
- Wheel center-to-center separation: 0.34 m
- Caster radius: 0.025 m
z_base_link = 0.05m + (0.15m/2) = 0.125m

## Poses relative to base link
| Component   |     (x) |     (y) |     (z) |
| ----------- | ------: | ------: | ------: |
| Left wheel  |  `0.00` | `+0.17` | `-0.05` |
| Right wheel |  `0.00` | `-0.17` | `-0.05` |
| Rear caster | `-0.18` |  `0.00` | `-0.10` |
| LiDAR       |  `0.05` |  `0.00` | `0.095` |
| IMU         |  `0.00` |  `0.00` |  `0.00` |
| Camera      |  `0.20` |  `0.00` | `0.075` |


## Sensors
- 2D LiDar
- RGB-D camera for later semantic exploration
- IMU for angular velocity and acceleration 
- Wheel encoders represented through simulated wheel-joint states

## Frame Hierarchy
base_footprint
└── base_link
    ├── left_wheel_link
    ├── right_wheel_link
    ├── caster_link
    ├── lidar_link
    ├── imu_link
    └── camera_link

## Fixed and Dynamic Joints
- base_footprint → base_link: fixed
- base_link → lidar_link: fixed
- base_link → imu_link: fixed
- base_link → camera_link: fixed
- base_link → caster_link: initially simplified as fixed
- base_link → left_wheel_link: continuous rotation
- base_link → right_wheel_link: continuous rotation
- odom → base_footprint: dynamic, published by odometry—not URDF
- map → odom: dynamic, published by SLAM—not URDF