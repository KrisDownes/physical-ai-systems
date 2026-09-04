"""Regression check for camera interference with the GPU lidar."""

from pathlib import Path
import xml.etree.ElementTree as ET


def test_camera_link_has_no_lidar_obstructing_geometry():
    """The camera mount must not create a false obstacle in the scan."""
    xacro = Path(__file__).parents[1] / 'urdf' / 'rover.urdf.xacro'
    root = ET.parse(xacro).getroot()
    camera = root.find("./link[@name='camera_link']")
    assert camera is not None
    assert camera.find('visual') is None
    assert camera.find('collision') is None
    optical = root.find("./link[@name='camera_optical_frame']")
    joint = root.find("./joint[@name='camera_optical_joint']")
    assert optical is not None
    assert joint.get('type') == 'fixed'
    assert joint.find('parent').get('link') == 'camera_link'
    assert joint.find('child').get('link') == 'camera_optical_frame'
    assert joint.find('origin').get('rpy') == '-1.57079632679 0 -1.57079632679'
    frame = root.find(
        "./gazebo[@reference='camera_link']/sensor/gz_frame_id")
    assert frame.text == 'camera_optical_frame'
