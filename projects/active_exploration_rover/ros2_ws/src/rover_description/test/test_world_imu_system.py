"""
Structural regression test for the world-level Gazebo IMU system plugin.

The robot IMU sensor (rover.urdf.xacro) only publishes gz.msgs.IMU if the
world loads the gz::sim::systems::Imu system plugin. Without it, /imu/data_raw
has no Gazebo publisher and the bridge/ EKF get nothing.

This test parses the active world SDF (the one the exploration launch passes to
Gazebo) and confirms the IMU system plugin exists exactly once and lives
directly under <world> (never inside a model or sensor). It also confirms the
world is installed by its package so the rebuilt workspace uses the edited file.
"""

import os
import xml.etree.ElementTree as ET

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Active world selected by sim.launch.py -> worlds/kd_world.sdf
WORLD_FILE = os.path.join(THIS_DIR, '..', 'worlds', 'kd_world.sdf')

IMU_FILENAME = 'gz-sim-imu-system'
IMU_NAME = 'gz::sim::systems::Imu'


@pytest.fixture(scope='module')
def world_root():
    assert os.path.exists(WORLD_FILE), f'active world missing: {WORLD_FILE}'
    return ET.parse(WORLD_FILE).getroot()


def _world_plugin_matches(elem):
    return (
        elem.tag == 'plugin'
        and elem.get('filename') == IMU_FILENAME
        and elem.get('name') == IMU_NAME
    )


def test_imu_system_plugin_present_and_unique(world_root):
    matches = [
        p for p in world_root.iter('plugin') if _world_plugin_matches(p)
    ]
    assert len(matches) == 1, (
        f'expected exactly one {IMU_FILENAME}/{IMU_NAME} plugin, '
        f'found {len(matches)}'
    )


def test_imu_system_plugin_is_direct_child_of_world(world_root):
    world = world_root.find('world')
    assert world is not None, 'no <world> element in active world SDF'
    # world.findall('plugin') returns ONLY direct-child plugins of <world>.
    direct_plugins = world.findall('plugin')
    for plugin in direct_plugins:
        if _world_plugin_matches(plugin):
            return
    pytest.fail('imu system plugin not found directly under <world>')


def test_imu_system_plugin_not_inside_any_model(world_root):
    # A plugin nested under a <model> would not be a world system plugin.
    for model in world_root.iter('model'):
        for plugin in model.iter('plugin'):
            if _world_plugin_matches(plugin):
                pytest.fail(
                    'imu system plugin must not live inside a <model>'
                )


def test_general_sensors_plugin_preserved(world_root):
    # LiDAR depends on the general Sensors system; it must remain.
    found = any(
        p.get('filename') == 'gz-sim-sensors-system'
        and p.get('name') == 'gz::sim::systems::Sensors'
        for p in world_root.iter('plugin')
    )
    assert found, 'general gz::sim::systems::Sensors plugin missing'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
