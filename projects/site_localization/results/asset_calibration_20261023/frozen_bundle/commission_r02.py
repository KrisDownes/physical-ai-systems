"""Freeze one development commissioning bundle; never reads needle labels/results."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from commissioned_gauge import calibrate, sha


def main(root, out):
    # All inputs are static printed marks / geometry / pixels, never needle truth.
    obs_path = root/'results/r02_geometry_diagnostic_20261023/image_observations.json'
    old_path = root/'results/real_gauge_feasibility_20261023/method_inputs.json'
    obs = json.loads(obs_path.read_text())
    original = next(r for r in json.loads(old_path.read_text()) if r['index']==6)
    source = root/'data/task_verification/paddle_audit/inventory_20261023'/obs['source_image']
    out.mkdir(parents=True, exist_ok=False)
    reference = Image.open(source).convert('RGB').crop(tuple(obs['crop_box_xyxy']))
    reference.save(out/'reference.png')
    crop_origin = np.array(obs['crop_box_xyxy'][:2])
    center = (np.array(original['center'])-crop_origin).tolist()
    ticks = [{'value':m['value'], 'point':((np.array(m['outer'])+m['inner'])/2).tolist()}
             for m in obs['marks'] if 5<=m['value']<=10]
    anchors = [m for m in ticks if m['value'] in [5,10]]
    config = {'version':'known-asset-local-calibration-v1',
        'asset_name':'R02 development reference; physical unit ID unverified',
        'units':'MPa','valid_range':[5,10], 'numerical_tolerance_MPa':.5,
        'printed_marks':anchors,'calibration':calibrate(center, anchors),
        'detector':{k:original[k] for k in ['radius','scale_start_deg','scale_sweep_deg','min_value','max_value']},
        'registration':{'ring_min':.82,'ring_max':.99,'ratio':.7,'ransac_px':3,
                        'min_matches':10,'min_inliers':8,'ransac_seed':20261023},
        'commissioning_assistance':{
            'operator_time_seconds':None,
            'source':'Reused assistant-marked visible5/10 stroke midpoints, legacy manual pivot/radius/crop, user-confirmed numbered scale and MPa units; no new human session',
            'pixel_mark_uncertainty':'Approximately +/-2px placement, not calibrated statistics',
            'pivot_uncertainty':'Existing +/-3px estimate, not surveyed',
            'needle_reference_used':False,
            'identity':'Unknown physical unit; matching cannot establish identity',
            'geometry':'Local unrectified reference chart; planar inter-image H assumes negligible pointer/dial parallax and lens distortion'},
        'input_hashes':{str(p.relative_to(root)):sha(p) for p in [obs_path,old_path,source]}}
    config['detector'].update(center=center, panel_shape=list(reference.size))
    (out/'commissioning.json').write_text(json.dumps(config,indent=2)+'\n')
    # Separate from runtime calibration: never consumed by CommissionedGauge.read().
    (out/'development_marks.json').write_text(json.dumps(ticks,indent=2)+'\n')
    frozen = root/'results/real_gauge_feasibility_20261023/frozen_reader.py'
    if sha(frozen) != 'f30fec60585e9ecd16ca141c28536bb2c7efded66155f720df8fe2483e7ddfff':
        raise ValueError('Original reader hash changed')
    shutil.copyfile(frozen, out/'frozen_reader.py')
    code = root/'research/task_verification'
    for name in ['commissioned_gauge.py','commission_r02.py','check_asset_calibration.py']:
        shutil.copyfile(code/name, out/name)
    shutil.copyfile(root/'docs/task_verification/asset_calibration_protocol.md', out/'protocol.md')
    manifest = {'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                'opencv':cv2.__version__, 'numpy':np.__version__,
                'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                'status':'Frozen before development checks; no independent validation cohort exists'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'bundle':str(out),'manifest_sha256':sha(out/'manifest.json'),
                      'calibration':config['calibration']},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    main(a.root,a.out)
