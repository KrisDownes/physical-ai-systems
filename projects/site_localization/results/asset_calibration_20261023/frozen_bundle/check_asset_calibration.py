"""Fixed software/development checks. No independent images or pressure truth.

Run only after commissioning bundle freeze. Exact generated H is evaluator-only.
The reader receives each image and its frozen bundle, not H or per-test geometry.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np

from commissioned_gauge import CommissionedGauge, angle, calibrate, sha, value_at


def main(bundle, out):
    out.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(1)
    reader = CommissionedGauge(bundle)
    cfg = reader.config
    observations = json.loads((bundle/'development_marks.json').read_text())
    center = np.array(cfg['detector']['center'], float)
    marks = cfg['printed_marks']
    checks = []
    for m in observations:
        predicted, inside = value_at(angle(m['point'], center), cfg['calibration'])
        checks.append({'printed_value': m['value'], 'mapped_value': predicted,
                       'error': predicted-m['value'], 'inside': inside,
                       'used_for_calibration': m['value'] in [5, 10]})
    unused = [r for r in checks if not r['used_for_calibration']]
    (out/'graduation_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    reference = reader.reference
    corners = np.float32([[0,0], [424,0], [424,424], [0,424]])
    target = np.float32([[40,28], [405,8], [438,413], [15,442]])
    exact = cv2.getPerspectiveTransform(corners, target)
    warped = cv2.warpPerspective(reference, exact, (480,480), borderValue=220)
    samples = [('self_registration', reference, np.eye(3)),
               ('synthetic_planar_warp', warped, exact),
               ('blank_negative', np.full((480,480), 220, np.uint8), None)]
    outcomes = []
    with (out/'attempts.jsonl').open('x') as ledger:
        for name, image, oracle in samples:
            cv2.imwrite(str(out/(name+'.png')), image)
            ledger.write(json.dumps({'case': name, 'event':'start'})+'\n'); ledger.flush()
            start = time.perf_counter()
            try:
                result = reader.read(image)
                execution = 'complete'
            except Exception as error:
                result = {'error': repr(error)}
                execution = 'infrastructure_or_implementation_failure'
            row = {'case':name, 'execution':execution, 'wall_s':time.perf_counter()-start,
                   'result':result, 'independent_observation':False}
            if oracle is not None and 'reference_to_query' in result:
                estimated = np.array(result['reference_to_query'])
                # Checkpoints not used by feature registration, in reference pixels.
                pts = np.float32([center]+[m['point'] for m in observations])
                true_query = cv2.perspectiveTransform(pts[None], oracle)
                back = cv2.perspectiveTransform(true_query, np.linalg.inv(estimated))[0]
                row['max_reference_checkpoint_error_px'] = float(np.max(np.linalg.norm(back-pts, axis=1)))
                row['oracle_reference_to_query_evaluation_only'] = oracle.tolist()
            outcomes.append(row)
            ledger.write(json.dumps(row)+'\n'); ledger.flush()
    # Audit geometry transport with a point on a fixed ray. No reading reference.
    theta = 190.0
    point = center + 100*np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    warped_points = cv2.perspectiveTransform(np.float32([[center, point]]), exact)[0]
    direct = angle(warped_points[1], warped_points[0])
    recovered = cv2.perspectiveTransform(warped_points[None], np.linalg.inv(exact))[0]
    recovered_angle = angle(recovered[1], recovered[0])
    sensitivity = []
    for dx,dy in itertools.product([-3, 0, 3], repeat=2):
        shifted = center+[dx,dy]
        recal = calibrate(shifted, marks)
        old_value, _ = value_at(angle(point, shifted), cfg['calibration'])
        new_value, _ = value_at(angle(point, shifted), recal)
        sensitivity.append({'dx':dx,'dy':dy,'fixed_mapping_value':old_value,'recalibrated_value':new_value})
    geometry = {'fixed_ray_deg':theta,'direct_query_angle_deg':direct,
                'inverse_warped_angle_deg':recovered_angle,
                'center_sensitivity':sensitivity,
                'interpretation':'Geometric software sensitivity only, not uncertainty bounds for physical pressure.'}
    (out/'geometry_checks.json').write_text(json.dumps(geometry, indent=2)+'\n')
    a,b,negative = [r['result'] for r in outcomes]
    both = a.get('decision') == b.get('decision') == 'accept'
    delta = abs(a['reading_value']-b['reading_value']) if both else None
    summary = {'independent_real_validation_observations':0,
               'software_image_cases':len(outcomes), 'outcomes':outcomes,
               'unused_mark_max_abs_error_MPa':max(abs(m['error']) for m in unused),
               'unused_mark_check_pass':all(abs(m['error'])<=.5 for m in unused),
               'synthetic_reading_delta_MPa':delta,
               'synthetic_reading_check_pass':both and delta<=.5,
               'synthetic_geometry_check_pass':outcomes[1].get('max_reference_checkpoint_error_px', float('inf'))<=3,
               'blank_check_pass':negative.get('decision')=='abstain',
               'manifest_sha256':sha(bundle/'manifest.json')}
    (out/'result.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='outcomes'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.bundle, args.out)
