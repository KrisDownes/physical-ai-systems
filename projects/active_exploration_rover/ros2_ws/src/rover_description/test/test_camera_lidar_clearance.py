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
