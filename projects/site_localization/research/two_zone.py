"""Occluded two-room development fixture with physically fixed mission targets.

World generation uses privileged geometry, as every simulator must. Runtime
localization/repair receives the analytic ORIGINAL prior and sensor data, never
the changed-world boxes. The cfg obstacle list is for the independent judge.
This is not a held-out environment or an original mapping method.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WALL_THICKNESS = .12
START = [-6.0, -2.0, 0.0]
GOALS = [[-6.0, -1.5, 0.0], [-3.0, 1.5, math.pi / 2],
         [3.0, 1.5, 0.0], [6.0, -1.5, -math.pi / 2]]
COMMISSIONED = [
    [-8.0, 0.0, .12, 6.12], [8.0, 0.0, .12, 6.12],
    [0.0, -3.0, 16.12, .12], [0.0, 3.0, 16.12, .12],
    # Left door y[-2.4,-1.0], right door y[1.0,2.4].
    [-1.0, -2.7, .12, .6], [-1.0, 1.0, .12, 4.0],
    [1.0, -1.0, .12, 4.0], [1.0, 2.7, .12, .6],
    [0.0, 0.0, .12, 2.2],
    # Nonidentical off-route commissioned furniture; symmetry is not forced.
    [-6.7, .5, .4, .6], [-2.1, 2.4, .5, .4],
    [6.7, .5, .4, .6], [2.1, -2.4, .5, .4],
]
CHANGED_STORAGE = [[7.45, 0.0, .12, 4.9],
                   [4.5, 2.45, 6.0, .12], [4.5, -2.45, 6.0, .12]]


def _box(world, name, box, height=1.0, floor=False):
    x, y, sx, sy = box
    model = ET.SubElement(world, 'model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    center_z = -.05 if floor else height / 2
    ET.SubElement(model, 'pose').text = f'{x} {y} {center_z} 0 0 0'
    link = ET.SubElement(model, 'link', name='link')
    for kind in ['collision', 'visual']:
        shape = ET.SubElement(ET.SubElement(link, kind, name=kind), 'geometry')
        ET.SubElement(ET.SubElement(shape, 'box'), 'size').text = f'{sx} {sy} {height}'


def _route():
    return [
        {'pose': copy.deepcopy(GOALS[0]), 'inspection_index': 0},
        {'pose': copy.deepcopy(GOALS[1]), 'inspection_index': 1},
        {'pose': [-3.0, -1.7, -math.pi / 2], 'inspection_index': None},
        {'pose': [-1.8, -1.7, 0.0], 'inspection_index': None},
        {'pose': [-.55, -1.7, math.pi / 2], 'inspection_index': None},
        {'pose': [-.55, 1.7, 0.0], 'inspection_index': None},
        {'pose': [1.8, 1.7, 0.0], 'inspection_index': None},
        {'pose': copy.deepcopy(GOALS[2]), 'inspection_index': 2},
        {'pose': copy.deepcopy(GOALS[3]), 'inspection_index': 3},
    ]


def make(out, condition, remedy):
    """Write cell.sdf, cell.json and original commissioned map.txt into out.

    The optional frame-only control translates the raster's coordinate frame;
    map-origin edit metadata supplies its conventional exact correction.
    Changed geometry never appears in map.txt, including acquired_map mode:
    that mode must replace the map with an actual acquired artifact in run.py.
    """
    if condition not in ['unchanged', 'frame_shift', 'physical_change']:
        raise ValueError(f'Unsupported condition: {condition}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = ET.parse(ROOT / 'sim/cell.sdf')
    world = source.getroot().find('world')
    rover = copy.deepcopy(world.find("model[@name='rover']"))
    if rover is None:
        raise ValueError('Reference rover model missing')
    # Preserve physics, global sensor/contact plugins, and rover internals.
    for model in list(world.findall('model')):
        world.remove(model)
    _box(world, 'floor', [0.0, 0.0, 20.0, 10.0], height=.1, floor=True)
    original = copy.deepcopy(COMMISSIONED)
    changes = copy.deepcopy(CHANGED_STORAGE) if condition == 'physical_change' else []
    for index, box in enumerate(original):
        _box(world, f'commissioned_obstacle_{index}', box)
    for index, box in enumerate(changes):
        _box(world, f'changed_storage_{index}', box)
    rover.find('pose').text = f'{START[0]} {START[1]} 0 0 0 {START[2]}'
    world.append(rover)
    ET.indent(source)
    source.write(out / 'cell.sdf', encoding='unicode')

    resolution = .05
    width, height = 340, 140
    origin_x, origin_y = -8.5, -3.5
    xx, yy = np.meshgrid(origin_x + np.arange(width) * resolution,
                         origin_y + np.arange(height) * resolution)
    grid = np.full((height, width), -1, dtype=np.int8)
    for x, y, sx, sy in original:
        grid[(abs(xx - x) <= sx / 2 + 1e-10) &
             (abs(yy - y) <= sy / 2 + 1e-10)] = 1
    map_transform = [.4, 0.0, 0.0] if condition == 'frame_shift' and remedy == 'metadata_alignment' else [0., 0., 0.]
    raster_origin_x = origin_x + (.4 if condition == 'frame_shift' else 0.)
    with (out / 'map.txt').open('w') as stream:
        stream.write(f'{width} {height} {resolution} {raster_origin_x} {origin_y}\n')
        np.savetxt(stream, grid, fmt='%d')
    route = _route()
    points = np.array([START[:2]] + [item['pose'][:2] for item in route])
    route_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    cfg = dict(
        split='development', layout='two_zone_dogleg', condition=condition,
        scenario='two_zone_' + condition, remedy=remedy,
        start=copy.deepcopy(START), goals=copy.deepcopy(GOALS), route=route,
        dwell_s=2, judge_position_m=.15, judge_yaw_rad=math.radians(10), max_sim_s=300,
        obstacles=original + changes, commissioned_obstacles=original,
        geometry_changed=bool(changes),
        changes=dict(type='right-room storage masking three unchanged exterior walls', boxes=changes),
        map_source='analytic original commissioned geometry; privileged initial prior only',
        sensor_noise='none', map_from_mission=map_transform,
        physical_target_frame='immutable commissioning mission frame',
        initialization='Known commissioned left dock; no subsequent truth input',
        operator_events=[dict(type='known_dock_initialization', count=1, human_seconds=None)],
        reset_count=1, repeat_attempt_index=0,
        zones=dict(unchanged_left=[-8., -1., -3., 3.], changed_right=[1., 8., -3., 3.]),
        inspection_zones=['unchanged_left', 'unchanged_left', 'changed_right', 'changed_right'],
        route_length_m=route_length,
        budget=dict(max_linear_speed_m_s=.25, ideal_translation_s=route_length / .25,
                    inspection_dwell_s=8, max_sim_s=300,
                    note='Translation lower bound excludes acceleration, rotations, settling and resets'),
        limitations=['development fixture', 'noiseless sensors', 'analytic commissioned prior',
                     'fixed preplanned clear transits; no obstacle replanning', 'no hardware validation'],
    )
    if condition == 'frame_shift':
        cfg.update(frame_change=[.4, 0., 0.],
                   frame_change_information='supplied raster-origin edit metadata, not estimated truth')
    (out / 'cell.json').write_text(json.dumps(cfg, indent=2) + '\n')
    return cfg


def extend_return_loop(cfg):
    """Append unchanged-room reinspection and dock return, without new geometry.

    Call after make() and before the caller persists its final cell.json. The
    original one-way fixture remains the default. Docking is a separate outcome
    from the six inspection stops; the independent judge must enforce it.
    """
    if cfg.get('layout') != 'two_zone_dogleg' or len(cfg['goals']) != 4:
        raise ValueError('Return extension requires an unextended two-zone mission')
    cfg['goals'].extend([copy.deepcopy(cfg['goals'][1]), copy.deepcopy(cfg['goals'][0])])
    cfg['inspection_zones'].extend(['unchanged_left', 'unchanged_left'])
    cfg['route'].extend([
        # Retrace the clear right-room diagonal, then the corridor in reverse.
        {'pose': [3.0, 1.5, math.pi], 'inspection_index': None},
        {'pose': [1.8, 1.7, math.pi], 'inspection_index': None},
        {'pose': [-.55, 1.7, -math.pi / 2], 'inspection_index': None},
        {'pose': [-.55, -1.7, math.pi], 'inspection_index': None},
        {'pose': [-1.8, -1.7, math.pi], 'inspection_index': None},
        {'pose': [-3.0, -1.7, math.pi / 2], 'inspection_index': None},
        {'pose': copy.deepcopy(cfg['goals'][4]), 'inspection_index': 4},
        {'pose': copy.deepcopy(cfg['goals'][5]), 'inspection_index': 5},
        {'pose': copy.deepcopy(cfg['start']), 'inspection_index': None},
    ])
    cfg['max_sim_s'] = 400
    cfg['return_loop'] = True
    cfg['dock_completion'] = dict(pose=copy.deepcopy(cfg['start']),
                                  position_tolerance_m=cfg['judge_position_m'],
                                  yaw_tolerance_rad=cfg['judge_yaw_rad'],
                                  stationary_dwell_s=cfg['dwell_s'],
                                  required_after_inspection_index=5,
                                  frame=cfg['physical_target_frame'])
    points = np.array([cfg['start'][:2]] + [item['pose'][:2] for item in cfg['route']])
    cfg['route_length_m'] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    cfg['budget'].update(ideal_translation_s=cfg['route_length_m'] / cfg['budget']['max_linear_speed_m_s'],
                         inspection_dwell_s=len(cfg['goals']) * cfg['dwell_s'],
                         dock_dwell_s=cfg['dwell_s'], max_sim_s=400)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--condition', choices=['unchanged', 'frame_shift', 'physical_change'], default='unchanged')
    parser.add_argument('--remedy', default='fixed')
    args = parser.parse_args()
    cfg = make(args.out, args.condition, args.remedy)
    print(json.dumps(dict(out=str(args.out), route_length_m=cfg['route_length_m'],
                          budget=cfg['budget']), indent=2))


if __name__ == '__main__':
    main()
