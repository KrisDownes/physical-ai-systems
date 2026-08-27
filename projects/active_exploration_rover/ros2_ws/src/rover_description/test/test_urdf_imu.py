"""
Structural regression test for the IMU link/sensor in rover.urdf.xacro.

Expands the xacro with the same `xacro` tool the launch uses, then parses the
resulting URDF XML (no ROS runtime required) to confirm:

* imu_link is declared
* imu_link is connected to base_link by a fixed joint (imu_joint)
* A Gazebo IMU sensor is attached to imu_link publishing /imu/data_raw at the
  correct frame
"""

import os
import subprocess
import xml.etree.ElementTree as ET

import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
XACRO_FILE = os.path.join(THIS_DIR, '..', 'urdf', 'rover.urdf.xacro')


def _expand_urdf():
    result = subprocess.run(
        ['xacro', XACRO_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture(scope='module')
def urdf_root():
    xml_text = _expand_urdf()
    return ET.fromstring(xml_text)


def test_imu_link_present(urdf_root):
    link_names = {link.get('name') for link in urdf_root.findall('link')}
    assert 'imu_link' in link_names


def test_imu_fixed_joint_to_base_link(urdf_root):
    for joint in urdf_root.findall('joint'):
        if joint.get('name') == 'imu_joint':
            assert joint.get('type') == 'fixed'
            assert joint.find('parent').get('link') == 'base_link'
            assert joint.find('child').get('link') == 'imu_link'
            return
    pytest.fail('imu_joint fixed joint not found')


def test_gazebo_imu_sensor_publishes_imu_raw(urdf_root):
    for gazebo in urdf_root.findall('gazebo'):
        if gazebo.get('reference') == 'imu_link':
            sensor = gazebo.find('sensor')
            assert sensor is not None
            assert sensor.get('type') == 'imu'
            assert sensor.get('name') == 'imu'
            topic = sensor.find('topic')
            assert topic is not None
            assert topic.text == '/imu/data_raw'
            frame = sensor.find('gz_frame_id')
            assert frame is not None
            assert frame.text == 'imu_link'
            update_rate = sensor.find('update_rate')
            assert update_rate is not None
            assert float(update_rate.text) == 100.0
            assert sensor.find('always_on').text == 'true'
            return
    pytest.fail('Gazebo IMU sensor on imu_link not found')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__]))
