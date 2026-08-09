import math

from rover_control.scan_processing import nearest_valid_range_in_sector

def test_returns_nearest_range_inside_forward_sector():
    ranges = [
        0.05,
        2.0,
        0.7,
        1.3,
        0.04
    ]

    result = nearest_valid_range_in_sector(
        ranges=ranges,
        angle_min=math.radians(-40),
        angle_increment=math.radians(20),
        range_min=0.1,
        range_max=10.0,
        sector_center=0.0,
        sector_half_width=math.radians(20),
    )

    assert result == 0.7

def test_ignores_invalid_ranges():
    ranges = [
        math.nan,
        math.inf,
        0.03,
        0.8,
        13.0,
    ]

    result = nearest_valid_range_in_sector(
        ranges=ranges,
        angle_min=math.radians(-20),
        angle_increment=math.radians(10),
        range_min=0.1,
        range_max=10.0,
        sector_center=0.0,
        sector_half_width=math.radians(20),
    )

    assert result == 0.8

def test_returns_none_when_no_valid_range_exists():
    ranges = [
        math.nan,
        math.inf,
        0.01,
        14.0,
        0.05,
    ]

    result = nearest_valid_range_in_sector(
        ranges=ranges,
        angle_min=math.radians(-20),
        angle_increment=math.radians(10),
        range_min=0.1,
        range_max=10.0,
        sector_center=0.0,
        sector_half_width=math.radians(20),
    )

    assert result is None