"""Freeze scenario audits and execute the unchanged runner once per case."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/moved_asset'))
from render import blocked_segments

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def verify(files):
    for p,h in files.items():assert sha(Path(p))==h,p

def freeze(root):
    path=root/'execution_freeze.json'
    if path.exists():raise FileExistsError(path)
    pipeline=load(root/'pipeline_freeze.json');scenarios=load(root/'scenario_freeze.json')
    verify(pipeline['files']);verify(scenarios['files']);verify(scenarios['evaluator_files'])
    checks=[];base=ROOT/'results/continuous_asset_20261025/scenarios/fresh/moved'
    for entry in load(root/'evaluator_preflight/feasibility.json')['cases']:
        case=entry['case'];folder=root/'templates'/case;scene=load(folder/'private/scene.json')
        world=ET.parse(folder/'private/world.sdf').getroot().find('world')
        for source in (base/'public').iterdir():assert sha(source)==sha(folder/'public'/source.name)
        assert sha(base/'controller_config.json')==sha(folder/'controller_config.json')
        eval_old=load(base/'private/evaluation.json');eval_new=load(folder/'private/evaluation.json')
        for k in eval_old:
            if k not in ['case','target_present','physical_obstacles']:assert eval_old[k]==eval_new[k],k
        for i,box in enumerate(scene['obstacles']):
            model=world.find(f"model[@name='partition_{i}']")
            pose=np.fromstring(model.find('pose').text,sep=' ')
            size=np.fromstring(model.find('.//collision/geometry/box/size').text,sep=' ')
            assert np.allclose([*pose[:2],*size[:2],size[2]],box)
        oldworld=ET.parse(base/'private/world.sdf').getroot().find('world')
        robot=next(m for m in oldworld.findall('model') if m.find('plugin') is not None)
        assert ET.tostring(robot)==ET.tostring(world.find(f"model[@name='{robot.get('name')}']"))
        record=dict(case=case,public_config_unchanged=True,robot_unchanged=True,physical_criteria_unchanged=True,geometry_consistent=True)
        if entry['target_present']:
            asset=next(a for a in scene['assets'] if a['id']==7)
            h=np.array(entry['witness_image_truth']['homography'])
            corners=np.array([[100,180,1],[339,180,1],[339,419,1],[100,419,1]])
            projected=corners@h.T;uv=projected[:,:2]/projected[:,2,None]
            size=np.array(scene['camera']['image_size']);fits=bool(np.all(uv>2)&np.all(uv<size-3))
            min_edge=float(np.linalg.norm(uv-np.roll(uv,1,axis=0),axis=1).min())
            yaw=asset['pose'][2];basis=np.array([np.sin(yaw),-np.cos(yaw),0.])
            center=np.array([*asset['pose'][:2],.8]);right=np.array(scene['gauge_center_uv'])
            samples=np.array([[u,v,1] for u in np.linspace(100,339,9) for v in np.linspace(180,419,9)])
            worldpoints=center+(samples[:,0]-right[0])[:,None]*scene['meters_per_texture_pixel']*basis+(samples[:,1]-right[1])[:,None]*scene['meters_per_texture_pixel']*np.array([0,0,-1])
            occluded=blocked_segments(np.array([*entry['witness_pose'][:2],.8]),worldpoints,scene['obstacles'])
            record.update(marker_fits=fits,marker_min_edge_px=min_edge,marker_surface_samples=81,marker_occluded_samples=int(occluded.sum()))
            assert fits and min_edge>=16 and not occluded.any()
        checks.append(record)
    save(root/'evaluator_preflight/consistency.json',checks)
    files={str(p):sha(p) for p in HERE.glob('*.py')}
    protocol=ROOT/'docs/connected_layout/protocol.md';files[str(protocol)]=sha(protocol)
    files.update(scenarios['files']);files.update(pipeline['files'])
    files.update({str(p):sha(p) for p in (root/'evaluator_preflight').iterdir() if p.is_file()})
    save(path,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,cases=pipeline['cases'],
        seed=pipeline['seed'],wall_limit_s=pipeline['wall_limit_s'],pipeline_freeze_sha256=sha(root/'pipeline_freeze.json'),
        scenario_freeze_sha256=sha(root/'scenario_freeze.json'),repeat_policy='one attempt per case; preserve all; no algorithmic reruns',
        physical_outcomes_opened=False))
    print('Execution freeze',sha(path))

def execute(root):
    manifest=load(root/'execution_freeze.json')
    for case in manifest['cases']:
        verify(manifest['files']);out=root/('mission_'+case)
        if out.exists():raise FileExistsError('Preserve attempted run: '+str(out))
        command=['/usr/bin/python3',str(ROOT/'research/continuous_asset/run.py'),'--template',str(root/'templates'/case),
                 '--out',str(out),'--seed',str(manifest['seed']),'--wall-limit',str(manifest['wall_limit_s'])]
        with (root/(case+'.log')).open('x') as f:code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Preserved execution/scoring error: '+case)
        result=load(out/'outcome.json');print(case,result['complete_contract'],result['termination'],result.get('inspection_reason'),flush=True)
        if result['infrastructure_failure']:raise RuntimeError('Preserved infrastructure-labelled failure: '+case+'; inspect cause before continuation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else freeze)(a.root.resolve())
