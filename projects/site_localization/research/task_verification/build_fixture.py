"""Counterfactual pinhole-camera evidence, NOT real images or new robot missions.

Truth is used here for sensor generation/evaluation, never by verify.py.
Run from the repository root; refuses to replace a previous experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def panel(fraction, tag_id=7, hide=None):
    im = np.full((700, 1000), 245, np.uint8)
    cv2.circle(im, (650, 350), 220, 30, 5)
    for i in range(51):
        angle = math.radians(135 + 270*i/50)
        points = [tuple(np.round([650+r*math.cos(angle), 350+r*math.sin(angle)]).astype(int))
                  for r in (198 if i % 5 == 0 else 210, 220)]
        cv2.line(im, *points, 15, 3)
        if i % 5 == 0:
            x, y = np.array([650, 350]) + 187*np.array([math.cos(angle), math.sin(angle)])
            cv2.putText(im, str(i//5), (int(x)-7, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, .6, 25, 2)
    angle = math.radians(135 + 270*fraction)
    tip = tuple(np.round([650+172*math.cos(angle), 350+172*math.sin(angle)]).astype(int))
    cv2.line(im, (650, 350), tip, 15, 7)
    cv2.circle(im, (650, 350), 12, 20, -1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    im[70:310, 70:310] = cv2.aruco.generateImageMarker(dictionary, tag_id, 240)
    if hide == 'needle':
        cv2.circle(im, (650, 350), 182, 245, -1)
    elif hide == 'tag':
        im[50:330, 50:330] = 245
    return im


def render(texture, pose, goal):
    """Image columns point camera-right; world SE(2) uses x-forward/y-left."""
    # Gauge centre is 1.5 m ahead of the fixed physical goal, not the estimate.
    gx, gy, yaw = goal
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([math.sin(yaw), -math.cos(yaw)])
    center = np.array([gx, gy]) + 1.5*forward
    x, y, heading = pose
    camera_forward = np.array([math.cos(heading), math.sin(heading)])
    camera_right = np.array([math.sin(heading), -math.cos(heading)])
    focal = 640/math.tan(math.radians(30))

    def project(uv):
        xy = center + (uv[0]-650)*.0005*right
        delta = xy - np.array([x, y])
        depth = float(delta @ camera_forward)
        assert depth > 0
        return [640 + focal*float(delta @ camera_right)/depth,
                360 + focal*(uv[1]-350)*.0005/depth]

    corners = np.float32([[0, 0], [999, 0], [999, 699], [0, 699]])
    target = np.float32([project(p) for p in corners])
    h = cv2.getPerspectiveTransform(corners, target)
    image = cv2.warpPerspective(texture, h, (1280, 720), borderValue=110)
    face = np.array([project(p) for p in [[430, 130], [870, 130], [870, 570], [430, 570]]])
    diameter = min(np.linalg.norm(face[1]-face[0]), np.linalg.norm(face[3]-face[0]))
    fits = bool(np.all(face >= [2, 2]) and np.all(face <= [1277, 717]))
    return image, {'face_corners': face.tolist(), 'diameter_px': float(diameter), 'fits': fits,
                   'panel_to_image': h.tolist()}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    (out/'inputs').mkdir()
    reference = panel(.5)
    cv2.imwrite(str(out/'inputs/reference.png'), reference)
    commissioning = {'reference': 'reference.png', 'intended_tag_id': 7, 'tag_family': '36h11',
        'panel_shape': [1000, 700], 'tag_corners': [[70, 70], [309, 70], [309, 309], [70, 309]],
        'center': [650, 350], 'radius': 220, 'scale_start_deg': 135, 'scale_sweep_deg': 270,
        'min_value': 0, 'max_value': 10, 'units': 'arbitrary_units',
        'assistance': 'Operator-approved rigid tag-to-dial geometry and unique ID; ideal calibration'}
    save(out/'inputs/commissioning.json', commissioning)
    cases = [('clear', .35, 7, None), ('changed_reading', .7, 7, None),
             ('needle_occluded', .35, 7, 'needle'), ('wrong_asset', .35, 8, None),
             ('tag_occluded', .35, 7, 'tag')]
    samples, truth, provenance = [], [], {}
    for policy in ['trimmed', 'capped']:
        src = ROOT/'results/review_matched_pair_20261023'/f'maintenance_bays__unchanged__{policy}__deployment'
        outcome = json.loads((src/'outcome.json').read_text())
        claim = outcome['claimed_stop_errors'][0]
        capture_time = claim['claim_time_s'] - .1
        states = [j for l in (src/'judge.jsonl').open() if (j := json.loads(l)).get('type') == 'truth']
        state = min(states, key=lambda r: abs(r['time']-capture_time))
        qx, qy, qz, qw = state['q']
        yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        pose = [state['x'], state['y'], yaw]
        goal = json.loads((src/'cell.json').read_text())['goals'][0]
        provenance[policy] = {'source': str(src.relative_to(ROOT)), 'claim': claim,
            'capture_requested_s': capture_time, 'capture_actual_s': state['time'], 'pose': pose,
            'goal': goal, 'original_mission_success': outcome['mission_success'],
            'files_sha256': {n: hashlib.sha256((src/n).read_bytes()).hexdigest()
                             for n in ['judge.jsonl', 'controller.jsonl', 'cell.json', 'outcome.json']}}
        for name, fraction, asset, hidden in cases:
            sid = f'c{len(samples):03d}'
            image, geometry = render(panel(fraction, asset, hidden), pose, goal)
            cv2.imwrite(str(out/f'inputs/{sid}.png'), image)
            samples.append({'id': sid, 'image': sid+'.png', 'controller_claimed_arrival': True})
            positive = asset == 7 and hidden != 'needle' and geometry['fits'] and geometry['diameter_px'] >= 80
            truth.append({'id': sid, 'source_group': policy, 'condition': name, 'asset_id': asset,
                          'needle_visible': hidden != 'needle', 'fraction': fraction,
                          'task_positive': positive, **geometry})
    save(out/'inputs/samples.json', samples)
    save(out/'evaluation_truth.json', truth)
    save(out/'sensor_generation_provenance.json', provenance)
    files = [ROOT/'docs/task_verification/experiment.md', *Path(__file__).parent.glob('*.py')]
    save(out/'pre_execution_manifest.json', {'protocol': str(files[0]),
         'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         'cv2': cv2.__version__, 'numpy': np.__version__, 'images': len(samples),
         'new_missions': 0, 'simulator': 'analytic camera retrofit, not Gazebo/physical camera',
         'observation_budget': 'one image, zero motion, no adaptive retries'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
