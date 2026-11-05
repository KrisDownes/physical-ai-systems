"""Bounded supplied/acquired map comparison; no inspection-rule changes."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD=ROOT/'results/moved_asset_20261024/v2'
import inspection_run as inspection
sys.path.insert(0,str(ROOT/'research'))
from correction_pipeline import anchor


def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    # Old commissioning supplies route/start only, never exact reference geometry.
    base=load(ROOT/'sim/cell.json')
    keys=['start','goals','dwell_s','judge_position_m','judge_yaw_rad']
    save(out/'commissioning.json',{k:base[k] for k in keys})
    for case in ['unchanged','moved','wrong_asset','absent']:
        shutil.copytree(OLD/'prepared'/(case+'__autonomous'),out/'templates'/case)
    files=[*HERE.glob('*.py'),*(p for p in (HERE/'reused_rover').rglob('*') if p.is_file() and '__pycache__' not in str(p)),ROOT/'results/autonomous_asset_20261024/perception_freeze.json',* (ROOT/'research/moved_asset').glob('*.py')]
    files += [ROOT/'research'/n for n in ['run.py','controller.py','frames.py','scenarios.py','score.py',
        'readiness.py','slam_bridge.py','slam_params.yaml','correction_pipeline.py','transfer_surface_diagnostic.py']]
    files += [ROOT/'sim/judge_node.py',ROOT/'sim/score_mission.py',ROOT/'vendor/offline_amcl',ROOT/'vendor/setup_amcl.bash',
        ROOT/'docs/autonomous_asset/protocol.md',out/'commissioning.json']
    bundle=ROOT/'results/asset_calibration_20261023/frozen_bundle'
    files += [p for p in bundle.iterdir() if p.is_file()]
    files += [p for p in (out/'templates').rglob('*') if p.is_file()]
    files += list({Path(a['texture']) for p in (out/'templates').glob('*/private/scene.json') for a in load(p)['assets']})
    (out/'source').mkdir()
    for p in HERE.glob('*.py'):shutil.copyfile(p,out/'source'/p.name)
    save(out/'freeze.json',dict(revision='autonomous-asset-v2',frozen_utc=now(),domain=215,
        acquisition_wall_limit=300,inspection_wall_limit=500,
        files={str(p):sha(p) for p in files},parent_freeze_sha256=sha(OLD/'freeze.json'),
        acquisition_order=['unchanged','moved'],inspection_order=['unchanged','moved','wrong_asset','absent'],
        map_lineage={'unchanged':'unchanged','moved':'moved','wrong_asset':'moved','absent':'moved'},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print('Frozen',out,sha(out/'freeze.json'))


def normalize(acquisition):
    """Dock anchor only. Axis-aligned raster translation is exact, no interpolation."""
    record=anchor(acquisition,[-2.,-2.,0.]);transform=record['target_from_source']
    if abs(transform[2])>1e-10:raise ValueError('Rotated dock gauge needs explicit raster/frame adapter; do not silently drop yaw')
    source=acquisition/'mapper/map.txt';target=acquisition/'map_in_site.txt'
    with source.open() as f:header=np.fromstring(f.readline(),sep=' ');body=f.read()
    header[3:5]+=transform[:2]
    target.write_text(f'{int(header[0])} {int(header[1])} {header[2]:.12g} {header[3]:.12g} {header[4]:.12g}\n'+body)
    save(acquisition/'map_transfer.json',dict(source=str(source),source_sha256=sha(source),
        target=str(target),target_sha256=sha(target),site_from_slam=transform,
        known_dock_assistance=True,registration_to_world_geometry=False,
        raster_cells_unchanged=True,operation='translate origin only; reject nonzero rotation'))
    return target


def no_other_server():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
        except (OSError,PermissionError):continue
        if any(Path(a).name=='gz' and args[i+1:i+3]==['sim','-s'] for i,a in enumerate(args)):
            raise RuntimeError('Another simulator is active; no launch: '+str(p))


def execute(out,stage):
    frozen=load(out/'freeze.json')
    for p,h in frozen['files'].items():assert sha(p)==h,p
    # Separate orchestration lock; existing acquisition runner takes domain lock itself.
    lock=open('/tmp/site_localization_autonomous_asset.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    no_other_server()
    def event(**values):
        with (out/'attempts.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),**values))+'\n')
    cases=['unchanged','moved'] if stage=='initial' else ['wrong_asset','absent']
    for case in cases:
      for treatment in ['commissioned','exploration']:
        mapping_case=frozen['map_lineage'][case];acq=out/('acquisition__'+mapping_case+'__'+treatment)
        if not acq.exists():
            command=['/usr/bin/python3',str(HERE/'acquire.py'),'--template',str(out/'templates'/case),
                '--commissioning',str(out/'commissioning.json'),'--out',str(acq),'--seed','7301',
                '--domain',str(frozen['domain']),'--wall-limit',str(frozen['acquisition_wall_limit'])]
            if treatment=='exploration':command.append('--explore')
            event(event='start',kind='acquisition',case=case,command=command,output=str(acq))
            before=time.monotonic()
            with (out/(acq.name+'.log')).open('x') as f:code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
            health=load(acq/'run.json') if (acq/'run.json').exists() else {}
            event(event='finish',kind='acquisition',case=case,output=str(acq),returncode=code,
                wall_s=time.monotonic()-before,run=health)
            if code or health.get('stop_reason')!='controller_completed' or not(acq/'mapper/map.txt').exists():
                raise RuntimeError('Acquisition failure retained; diagnose before inspection')
            normalize(acq)
        if not(acq/'map_in_site.txt').exists():raise RuntimeError('Missing valid acquisition transfer; no bypass')
        if True:
            name=case+'__'+treatment;prepared=out/'prepared'/name;destination=out/name
            if destination.exists():raise FileExistsError('Preserve original attempt '+str(destination))
            shutil.copytree(out/'templates'/case,prepared)
            source_map=acq/'map_in_site.txt'
            shutil.copyfile(source_map,prepared/'map.txt')
            lineage=dict(treatment=treatment,source=str(source_map),source_sha256=sha(source_map),
                mapping_episode=str(acq),
                mapping_cost_not_in_inspection=True,frame='commissioned site',map_from_mission=[0,0,0])
            save(prepared/'map_lineage.json',lineage)
            effective={str(p.relative_to(prepared)):sha(p) for p in prepared.rglob('*') if p.is_file()}
            event(event='start',kind='inspection',case=case,treatment=treatment,output=str(destination),
                effective_hashes=effective,map_lineage=lineage)
            no_other_server()
            # Same frozen corrected perception in both arms; remaining inspection logic unchanged.
            record=inspection.run_one(dict(prepared=str(prepared),output=str(destination)),
                dict(domain=frozen['domain'],wall_guard_s=frozen['inspection_wall_limit']))
            event(event='finish',kind='inspection',case=case,treatment=treatment,output=str(destination),run=record)
            print(name,record['stop_reason'],'score',record['score_returncode'],flush=True)
            if record['stop_reason']!='policy_finished' or record['score_returncode']:
                raise RuntimeError('Inspection infrastructure failure preserved; stop for diagnosis')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',choices=['initial','extension']);a=p.parse_args()
    if a.execute:execute(a.out.resolve(),a.execute)
    else:prepare(a.out.resolve())
