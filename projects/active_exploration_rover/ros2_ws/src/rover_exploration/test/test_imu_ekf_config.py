"""
Structural regression tests for the IMU-assisted EKF odometry pipeline.

These tests parse the EKF YAML, the Gazebo bridge launch, and the exploration
launch directly (no ROS runtime required) to lock in the pose-graph fixes that
prevent the rotated/duplicated SLAM map:

* The EKF owns odom -> base_footprint (world_frame == odom, base_link_frame ==
  base_footprint, publish_tf true, two_d_mode true).
* Wheel odometry fuses only forward velocity (no wheel yaw, no wheel yaw rate).
* The IMU fuses yaw rate only (orientation yaw is not fused) per the selected
  valid mode.
* The active bridge carries /imu/data_raw, keeps /model/kd_bot/odometry, and no
  longer republishes /model/kd_bot/tf onto /tf.
* The exploration launch starts robot_localization/ekf_node (not gated on
  enable_motion).
"""

import importlib.util
import os

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# test file: <ws>/src/rover_exploration/test/test_imu_ekf_config.py
ROVER_EXPLORATION_DIR = os.path.dirname(HERE)                 # <ws>/src/rover_exploration
SRC_DIR = os.path.dirname(ROVER_EXPLORATION_DIR)              # <ws>/src


def _load_yaml(rel_path):
    path = os.path.join(ROVER_EXPLORATION_DIR, rel_path)
    with open(path, 'r') as handle:
        return yaml.safe_load(handle), path


def _load_launch_source(rel_path):
    """Return the module object for a launch file without importing ROS."""
    path = os.path.join(SRC_DIR, rel_path)
    spec = importlib.util.spec_from_file_location('launch_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_bridge_arguments(launch_module):
    """Pull every bridge topic string from the parameter_bridge Node arguments."""
    topics = []
    desc = launch_module.generate_launch_description()
    for entity in desc.entities:
        # Bridge Node is a launch_ros Node action. Its forwarded Gazebo topic
        # strings live in the private _Node__arguments list.
        arguments = getattr(entity, '_Node__arguments', None)
        if isinstance(arguments, (list, tuple)):
            topics.extend(a for a in arguments if isinstance(a, str) and '@' in a)
        remappings = getattr(entity, '_Node__remappings', None)
        if isinstance(remappings, (list, tuple)):
            topics.extend(
                (r[0], r[1]) for r in remappings if isinstance(r, (list, tuple))
                and len(r) == 2
            )
    return topics


def _iter_nodes(launch_module):
    """Yield (package, executable, condition) for every launch_ros Node."""
    desc = launch_module.generate_launch_description()
    for entity in desc.entities:
        package = getattr(entity, '_Node__package', None)
        executable = getattr(entity, '_Node__node_executable', None)
        if package is None and executable is None:
            continue
        condition = getattr(entity, '_Action__condition', None)
        yield package, executable, condition


# ---------------------------------------------------------------------------
# EKF configuration
# ---------------------------------------------------------------------------


def test_ekf_required_frames_and_publish():
    params, _ = _load_yaml('config/ekf.yaml')
    node_params = params['ekf_filter_node']['ros__parameters']
    assert node_params['world_frame'] == 'odom'
    assert node_params['odom_frame'] == 'odom'
    assert node_params['base_link_frame'] == 'base_footprint'
    assert node_params['publish_tf'] is True
    assert node_params['two_d_mode'] is True


def test_ekf_fuses_only_forward_wheel_velocity():
    params, _ = _load_yaml('config/ekf.yaml')
    node_params = params['ekf_filter_node']['ros__parameters']
    # 15-element order: x,y,z, roll,pitch,yaw, vx,vy,vz,
    #                   vroll,vpitch,vyaw, ax,ay,az
    odom_config = node_params['odom0_config']
    assert len(odom_config) == 15
    assert odom_config[6] is True   # vx (forward velocity) enabled
    assert odom_config[7] is False  # vy disabled
    assert odom_config[8] is False  # vz disabled
    # Wheel yaw must NOT be fused.
    assert odom_config[5] is False  # yaw
    # Wheel yaw rate must NOT be fused.
    assert odom_config[11] is False  # vyaw
    # Wheel pose x/y must NOT be fused.
    assert odom_config[0] is False  # x
    assert odom_config[1] is False  # y


def test_ekf_imu_fuses_yaw_rate_only():
    params, _ = _load_yaml('config/ekf.yaml')
    node_params = params['ekf_filter_node']['ros__parameters']
    imu_config = node_params['imu0_config']
    assert len(imu_config) == 15
    # Active mode: yaw-rate-only. Orientation yaw is NOT fused.
    assert imu_config[5] is False   # yaw (orientation) not fused
    assert imu_config[11] is True  # vyaw (yaw rate)
    assert node_params['imu0_relative'] is False
    assert node_params['imu0_differential'] is False
    # Linear acceleration not fused.
    assert imu_config[12] is False  # ax
    assert imu_config[13] is False  # ay
    assert imu_config[14] is False  # az


def test_ekf_inputs_exclude_ground_truth():
    params, _ = _load_yaml('config/ekf.yaml')
    node_params = params['ekf_filter_node']['ros__parameters']
    assert node_params['odom0'] == '/model/kd_bot/odometry'
    assert node_params['imu0'] == '/imu/data_raw'
    # Ground truth must never feed the EKF.
    assert '/ground_truth/odometry' not in (
        node_params.get('odom0'),
        node_params.get('imu0'),
    )
    for key in node_params:
        assert 'ground_truth' not in str(key)


# ---------------------------------------------------------------------------
# Gazebo bridge configuration
# ---------------------------------------------------------------------------


def test_bridge_carries_imu_and_wheel_odom_only():
    launch_module = _load_launch_source(
        'rover_description/launch/display.launch.py'
    )
    topics = _collect_bridge_arguments(launch_module)
    topic_strings = [t for t in topics if isinstance(t, str)]
    assert any(t.startswith('/imu/data_raw@') for t in topic_strings)
    assert any(
        t.startswith('/model/kd_bot/odometry@') for t in topic_strings
    )
    # Wheel odometry must be retained.
    assert not any(
        t.startswith('/model/kd_bot/tf@') for t in topic_strings
    )
    # /tf remapping from Gazebo must be gone.
    remappings = [t for t in topics if isinstance(t, tuple)]
    assert ('/model/kd_bot/tf', '/tf') not in remappings
    # Ground truth is still bridged for diagnostics only.
    assert any(
        t.startswith('/ground_truth/odometry@') for t in topic_strings
    )


# ---------------------------------------------------------------------------
# Exploration launch wiring
# ---------------------------------------------------------------------------


def test_exploration_launch_starts_ekf_node():
    launch_module = _load_launch_source('rover_exploration/launch/exploration.launch.py')
    found = False
    for package, executable, condition in _iter_nodes(launch_module):
        if package == 'robot_localization' and executable == 'ekf_node':
            found = True
            # Must not be gated on enable_motion.
            assert condition is None
    assert found, 'robot_localization/ekf_node not present in exploration launch'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
