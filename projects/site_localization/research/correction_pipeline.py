"""Known-dock frame preservation and conventional surface registration audit."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from frames import compose,inverse
from transfer_surface_diagnostic import boundary_map

ROOT=Path(__file__).resolve().parents[1]


def anchor(acquisition,dock):
    acquisition=Path(acquisition)
    with (acquisition/'mapper/mapper_pose.csv').open() as stream:first=next(csv.DictReader(stream),None)
    if first is None:raise ValueError('No initial mapper pose; cannot establish dock frame')
    pose=np.array([float(first[k]) for k in ['x','y','yaw']]);stamp=float(first['timestamp'])
    if not np.isfinite(pose).all() or stamp>1 or np.linalg.norm(pose[:2])>.02 or abs(pose[2])>.02:
        raise ValueError('Initial mapper pose does not verify the declared stationary dock gauge')
    value=compose(np.array(dock),inverse(pose))
    record=dict(target_from_source=value.tolist(),map_from_mission=inverse(value).tolist(),
        known_site_from_dock=list(dock),first_mapper_pose=pose.tolist(),first_mapper_stamp_s=stamp,
        verified=True,truth_used=False,source='commissioned dock correspondence and official initial mapper pose',
        assumptions=['dock physically repeatable and known in mission coordinates',
            'first fixed SLAM node at stationary dock; graph gauge preserved across optimization',
            'continued/localization reuse this graph frame, never independently refit it'])
    (acquisition/'dock_anchor.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def surface_audit(acquisition,dock):
    acquisition=Path(acquisition);start=time.monotonic()
    source=acquisition/'source_surface.txt';target=acquisition/'target_surface.txt'
    extraction=dict(source=boundary_map(acquisition/'mapper/map.txt',source),
                    target=boundary_map(acquisition/'map.txt',target))
    (acquisition/'surface_extraction.json').write_text(json.dumps(extraction,indent=2)+'\n')
    output=acquisition/'surface_alignment.json'
    command=[sys.executable,str(ROOT/'research/register_maps.py'),'--source',str(source),'--target',str(target),
        '--output',str(output),'--prior',*map(str,dock),'--method','robust','--selection-objective','trimmed',
        '--max-correspondence','.15','--trim-fraction','.7','--huber-delta','.08','--inlier-distance','.12',
        '--translation-radius','.25','--yaw-radius','.1','--max-points','12000','--score-cap','.3',
        '--iterations','100','--translation-tolerance','.00001','--yaw-tolerance','.00001']
    reply=subprocess.run(command,capture_output=True,text=True,timeout=90)
    (acquisition/'surface_registration.stdout').write_text(reply.stdout)
    (acquisition/'surface_registration.stderr').write_text(reply.stderr)
    result=dict(returncode=reply.returncode,wall_s=time.monotonic()-start,command=command,
        frame_policy='Dock correspondence authoritative; free surface ICP is diagnostic only',
        registration_source_sha256=hashlib.sha256((ROOT/'research/register_maps.py').read_bytes()).hexdigest())
    (acquisition/'registration_cost.json').write_text(json.dumps(result,indent=2)+'\n')
    if reply.returncode:raise RuntimeError('Surface audit could not produce a fit')
    return output
