"""Prospective transfer fixtures: simulator geometry is never a repair input.

The analytic ORIGINAL map is a privileged historical commissioning prior.
Changed geometry is used only to instantiate the world and physical judge.
No result-dependent geometry, noise or controller parameters are generated.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ('maintenance_bays', 'l_workshop')
CONDITIONS = ('unchanged', 'additions', 'rearrangement')


def _spec(layout):
    if layout == 'maintenance_bays':
        walls = [[-7, 0, .12, 8.12], [7, 0, .12, 8.12],
                 [0, -4, 14.12, .12], [0, 4, 14.12, .12],
                 [-1.95, 1.3, 10.1, .12], [5.95, 1.3, 2.1, .12],
                 [-1.45, -1.3, 11.1, .12], [6.45, -1.3, 1.1, .12]]
        furniture = [[5.8, 2.8, .7, 1.2], [1, 3.1, 1, .6], [2, -3.1, 1, .6]]
        additions = [[4, 3.5, 3, .35], [6.35, 2.7, .3, 1.3], [3.5, -3.5, 4, .35]]
        rearranged = [[5.8, 2.0, .7, 1.2], [1, 3.1, 1, .6], [1, -2.4, 1, .6]]
        start = [-6, -.3, 0]
        fixtures = [[-5, 0, 0], [-2, 0, 0], [2.5, 2.6, math.pi/2], [5, -2.6, -math.pi/2]]
        # Visits 0 and 1 are repeated after both maintenance bays.
        path = [(fixtures[0], 0), (fixtures[1], 1), ([4, 0, 0], None),
                ([4, 2.6, math.pi/2], None), (fixtures[2], 2),
                ([4, 2.6, -math.pi/2], None), ([4, 0, 0], None),
                ([5, 0, -math.pi/2], None), (fixtures[3], 3),
                ([5, 0, math.pi], None), (fixtures[1], 4), (fixtures[0], 5), (start, None)]
        names = ['corridor utility panel', 'corridor emergency station', 'upper-bay service fitting', 'lower-bay service fitting']
        workflow = 'Maintenance corridor with two service bays: added perimeter stores or relocated cabinets after maintenance.'
        bounds = [-7, 7, -4, 4]
    elif layout == 'l_workshop':
        # A solid excluded wing produces an actual L-shaped traversable space.
        walls = [[-6, 0, .12, 8.12], [6, 0, .12, 8.12],
                 [0, -4, 12.12, .12], [0, 4, 12.12, .12], [3.5, 2, 5, 4]]
        furniture = [[-5.3, -2.5, .8, 1], [-4, .1, 1, .6], [-.1, 3.2, 1, .6],
                     [2, -3.4, 1.2, .4], [5.4, -2, .4, 1]]
        additions = [[1.8, -3.5, 3, .25], [5.55, -1.8, .3, 2.8], [3, -.35, 3.5, .3]]
        rearranged = [[-5.3, -2.5, .8, 1], [-4, .1, 1, .6], [-.1, 3.2, 1, .6],
                      [.2, -3.4, 1.2, .4], [4.8, -3.4, 1, .4]]
        start = [-4.8, 2.8, 0]
        fixtures = [[-4.7, 1, math.pi], [-1, 2.5, math.pi/2], [4.7, -1.1, 0], [3.2, -2.8, -math.pi/2]]
        path = [(fixtures[0], 0), (fixtures[1], 1), ([-1, -1.1, 0], None),
                (fixtures[2], 2), (fixtures[3], 3), ([-3.5, -2.8, math.pi], None),
                ([-3.5, -.8, 0], None), ([-1, -.8, math.pi/2], None),
                (fixtures[1], 4), (fixtures[0], 5), (start, None)]
        names = ['upper-wing electrical panel', 'upper-wing alarm station', 'workshop pump gauge', 'workshop fixed service outlet']
        workflow = 'L-shaped workshop: work-in-progress storage along workshop walls or relocation of mobile cabinets.'
        bounds = [-6, 6, -4, 4]
    else:
        raise ValueError(layout)
    return dict(walls=walls, furniture=furniture, additions=additions, rearranged=rearranged,
                start=start, fixtures=fixtures, path=path, names=names, workflow=workflow, bounds=bounds)


def _point_segment(p, a, b):
    v = b-a
    t = np.clip(np.dot(p-a, v)/np.dot(v, v), 0, 1) if np.dot(v, v) else 0
    return float(np.linalg.norm(p-(a+t*v)))


def _segment_box_distance(a, b, box):
    """Exact Euclidean centerline-to-closed-axis-box distance (2-D)."""
    center, half = np.array(box[:2]), np.array(box[2:])/2
    lo, hi = center-half, center+half
    t0, t1 = 0., 1.
    for k in range(2):
        d = b[k]-a[k]
        if abs(d) < 1e-12:
            if a[k] < lo[k] or a[k] > hi[k]:
                break
        else:
            u, v = sorted(((lo[k]-a[k])/d, (hi[k]-a[k])/d))
            t0, t1 = max(t0, u), min(t1, v)
            if t0 > t1:
                break
    else:
        return 0.
    corners = [np.array([x, y]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    ends = [float(np.linalg.norm(p-np.clip(p, lo, hi))) for p in (a, b)]
    return min(ends + [_point_segment(p, a, b) for p in corners])


def audit(cfg):
    """Geometric feasibility, not a prediction of closed-loop success."""
    points = [cfg['start'][:2]] + [v['pose'][:2] for v in cfg['route']]
    distances = [min(_segment_box_distance(np.array(a), np.array(b), box)
                     for box in cfg['obstacles']) for a, b in zip(points, points[1:])]
    result = dict(inflation_radius_m=.29, segment_centerline_clearance_m=distances,
                  min_inflated_clearance_m=min(distances)-.29,
                  criterion='every segment has >0.08 m clearance after 0.29 m circular inflation',
                  scope='all planned straight segments and in-place rotations; not controller tracking guarantee')
    if result['min_inflated_clearance_m'] <= .08:
        raise ValueError(f'Infeasible route: {result}')
    return result


def _box(world, name, box, height=1., floor=False):
    x, y, sx, sy = box
    model = ET.SubElement(world, 'model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{x} {y} {-.05 if floor else height/2} 0 0 0'
    link = ET.SubElement(model, 'link', name='link')
    for kind in ('collision', 'visual'):
        geometry = ET.SubElement(ET.SubElement(link, kind, name=kind), 'geometry')
        ET.SubElement(ET.SubElement(geometry, 'box'), 'size').text = f'{sx} {sy} {height}'


def make(out, layout, condition, remedy):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    spec = _spec(layout)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    original = spec['walls'] + spec['furniture']
    actual = spec['walls'] + (spec['rearranged'] if condition == 'rearrangement' else spec['furniture'])
    if condition == 'additions':
        actual = actual + spec['additions']
    goals = copy.deepcopy(spec['fixtures'] + [spec['fixtures'][1], spec['fixtures'][0]])
    route = [dict(pose=copy.deepcopy(p), inspection_index=i) for p, i in spec['path']]
    # Transit headings face the next segment; fixture headings remain physical.
    for i, item in enumerate(route[:-1]):
        if item['inspection_index'] is None:
            dx, dy = np.array(route[i+1]['pose'][:2])-item['pose'][:2]
            item['pose'][2] = math.atan2(dy, dx)
    points = np.array([spec['start'][:2]] + [v['pose'][:2] for v in route])
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    cfg = dict(split='held_out_transfer', layout=layout, condition=condition,
               scenario=f'{layout}_{condition}', remedy=remedy,
               start=copy.deepcopy(spec['start']), goals=goals, route=route,
               obstacles=copy.deepcopy(actual), commissioned_obstacles=copy.deepcopy(original),
               geometry_changed=condition != 'unchanged', changes=dict(type=condition, workflow=spec['workflow']),
               map_source='analytic original commissioned geometry; privileged historical prior only',
               physical_target_frame='immutable commissioning mission frame', map_from_mission=[0., 0., 0.],
               initialization='Known commissioned dock; no subsequent truth input',
               operator_events=[dict(type='known_dock_initialization', count=1, human_seconds=None)],
               reset_count=1, repeat_attempt_index=0, sensor_noise='none',
               dwell_s=2, judge_position_m=.15, judge_yaw_rad=math.radians(10), max_sim_s=500,
               inspection_zones=['unchanged', 'unchanged', 'changed_sensitive', 'changed_sensitive', 'unchanged', 'unchanged'],
               return_loop=True, dock_completion=dict(pose=copy.deepcopy(spec['start']), position_tolerance_m=.15,
                   yaw_tolerance_rad=math.radians(10), stationary_dwell_s=2, required_after_inspection_index=5,
                   frame='immutable commissioning mission frame'),
               route_length_m=length, budget=dict(max_linear_speed_m_s=.25, ideal_translation_s=length/.25,
                   inspection_dwell_s=12, dock_dwell_s=2, max_sim_s=500),
               limitations=['noiseless simulation', 'analytic historical map', 'fixed preplanned transits',
                            'no physical transfer claim', 'operator time unmeasured'])
    cfg['geometry_audit'] = audit(cfg)
    tree = ET.parse(ROOT/'sim/cell.sdf')
    world = tree.getroot().find('world')
    rover = copy.deepcopy(world.find("model[@name='rover']"))
    for model in list(world.findall('model')):
        world.remove(model)
    _box(world, 'floor', [0, 0, 20, 14], height=.1, floor=True)
    for i, box in enumerate(actual):
        _box(world, f'fixture_obstacle_{i}', box)
    x, y, yaw = spec['start']
    rover.find('pose').text = f'{x} {y} 0 0 0 {yaw}'
    world.append(rover)
    ET.indent(tree)
    tree.write(out/'cell.sdf', encoding='unicode')
    x0, x1, y0, y1 = spec['bounds']
    resolution = .05
    origin_x, origin_y = x0-.5, y0-.5
    width, height = round((x1-x0+1)/resolution), round((y1-y0+1)/resolution)
    xx, yy = np.meshgrid(origin_x+np.arange(width)*resolution, origin_y+np.arange(height)*resolution)
    grid = np.full((height, width), -1, dtype=np.int8)
    for x, y, sx, sy in original:
        grid[(abs(xx-x) <= sx/2+1e-10) & (abs(yy-y) <= sy/2+1e-10)] = 1
    with (out/'map.txt').open('w') as stream:
        stream.write(f'{width} {height} {resolution} {origin_x} {origin_y}\n')
        np.savetxt(stream, grid, fmt='%d')
    manifest = dict(layout=layout, condition=condition, workflow=spec['workflow'], bounds=spec['bounds'],
                    fixtures=[dict(name=name, pose=p, meaning='fixed asset, unchanged across conditions')
                              for name, p in zip(spec['names'], spec['fixtures'])],
                    commissioned_obstacles=original, actual_obstacles=actual,
                    additions=spec['additions'], rearranged_furniture=spec['rearranged'],
                    route=route, route_length_m=length, geometry_audit=cfg['geometry_audit'],
                    provenance='prospectively specified fixture; geometry author did not inspect ROS outcomes',
                    judge_only_fields=['actual_obstacles', 'geometry_audit'],
                    historical_prior_privileged=True)
    (out/'fixture_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'cell.json').write_text(json.dumps(cfg, indent=2)+'\n')
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--layout', required=True, choices=LAYOUTS)
    parser.add_argument('--condition', required=True, choices=CONDITIONS)
    parser.add_argument('--remedy', default='fixed')
    args = parser.parse_args()
    cfg = make(args.out, args.layout, args.condition, args.remedy)
    print(json.dumps(dict(layout=cfg['layout'], condition=cfg['condition'], route_length_m=cfg['route_length_m'],
                          geometry_audit=cfg['geometry_audit']), indent=2))
