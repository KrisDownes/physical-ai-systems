"""Evaluation-only construction and geometric feasibility; no policy modification."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research/moved_asset'))
from planning import GridPlanner
from render import render, blocked_segments


def load(p): return json.loads(p.read_text())
def save(p, x): p.write_text(json.dumps(x, indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def geometry():
    return [[-3.2,0,.15,6.55,1.], [3.2,0,.15,6.55,1.],
            [0,-3.2,6.55,.15,1.], [0,3.2,6.55,.15,1.],
            [.5,-1.15,.18,4.1,1.], [.5,2.85,.18,.7,1.],
            [-2.35,.5,1.7,.18,1.]]


def floor_grid(boxes, radius=.23):
    axis=np.arange(-3.175,3.2,.05); yy,xx=np.meshgrid(axis,axis,indexing='ij')
    points=np.c_[xx.ravel(),yy.ravel()]
    clearance=np.full(len(points),np.inf)
    for x,y,sx,sy,_ in boxes:
        delta=np.maximum(abs(points-[x,y])-[sx/2,sy/2],0)
        clearance=np.minimum(clearance,np.linalg.norm(delta,axis=1))
    free=(clearance>radius).reshape(xx.shape)
    components,n=label(free)
    ix=np.argmin(abs(axis+2)); connected=components==components[ix,ix]
    return axis,points,free,connected,clearance


def build(root):
    frozen=load(root/'pipeline_freeze.json')
    for p,h in frozen['files'].items(): assert sha(Path(p))==h,p
    templates=root/'templates';templates.mkdir(exist_ok=False)
    oracle=root/'evaluator_preflight';oracle.mkdir()
    source=ROOT/'results/continuous_asset_20261025/scenarios/fresh/moved'
    boxes=geometry();axis,points,free,connected,clearance=floor_grid(boxes)
    # Truth occupancy map stays evaluator-only, never copied into a template.
    occupied=(clearance==0).reshape(free.shape)
    gp=oracle/'truth_grid.txt'
    with gp.open('w') as f:
        f.write(f'{len(axis)} {len(axis)} .05 {axis[0]} {axis[0]}\n')
        np.savetxt(f,np.where(occupied,1,-1),fmt='%d')
    planner=GridPlanner(gp,.52)
    fixed=dict(visible=[0,-1.5,0],moved=[2.45,-1.5,0],occluded=[-2.4,1.7,math.pi],
               wrong_asset=[2.45,-1.5,0],absent=None)
    records=[]
    for case in frozen['cases']:
        dst=templates/case;dst.mkdir();(dst/'private').mkdir()
        shutil.copytree(source/'public',dst/'public')
        shutil.copyfile(source/'controller_config.json',dst/'controller_config.json')
        scene=load(source/'private/scene.json');scene['obstacles']=boxes;scene['assets']=[]
        for identity,pose in [(7,fixed[case]),(8,[0,-1.5,0] if case in ['wrong_asset','absent'] else None)]:
            if pose is not None:
                scene['assets'].append(dict(id=identity,pose=pose,texture=str(ROOT/f'results/moved_asset_20261024/v2/textures/asset_{identity}.png')))
        save(dst/'private/scene.json',scene)
        evaluation=load(source/'private/evaluation.json');evaluation.update(case=case,target_present=fixed[case] is not None,physical_obstacles=[b[:4] for b in boxes])
        save(dst/'private/evaluation.json',evaluation)
        tree=ET.parse(source/'private/world.sdf');world=tree.getroot().find('world')
        sample=copy.deepcopy(world.find("model[@name='obstacle_0']"))
        for model in list(world.findall('model')):
            if model.get('name') not in ['floor','rover']: world.remove(model)
        # Preserve the existing robot exactly; its actual name is checked below.
        if world.find("model[@name='rover']") is None:
            original=ET.parse(source/'private/world.sdf').getroot().find('world')
            robot=next(m for m in original.findall('model') if m.find('plugin') is not None)
            world.append(copy.deepcopy(robot))
        for i,(x,y,sx,sy,h) in enumerate(boxes):
            model=copy.deepcopy(sample);model.set('name',f'partition_{i}');model.find('pose').text=f'{x} {y} {h/2} 0 0 0'
            for size in model.findall('.//box/size'):size.text=f'{sx} {sy} {h}'
            world.append(model)
        for asset in scene['assets']:
            x,y,yaw=asset['pose']
            panel=ET.fromstring(f'<model name="asset_panel_{asset["id"]}"><static>true</static><pose>{x} {y} .8 0 0 {yaw}</pose><link name="panel"><collision name="panel"><geometry><box><size>.01 .667 .4</size></box></geometry></collision></link></model>')
            world.append(panel)
        tree.write(dst/'private/world.sdf',encoding='unicode')
        record=dict(case=case,target_present=fixed[case] is not None,
                    connected_floor_area_m2=float(connected.sum()*.05**2),all_footprint_free_space_connected=bool(np.all(connected[free])),
                    policy_clearance_connected_cells=int(np.sum(np.isfinite(planner.paths_from([-2,-2])[0]))))
        if fixed[case] is not None:
            x,y,yaw=fixed[case];witness=[x-1.1*math.cos(yaw),y-1.1*math.sin(yaw),yaw]
            path=planner.path([-2,-2],witness)
            image,metadata=render(witness,scene);target=next(a for a in metadata['assets'] if a['actual_id']==7)
            cv2.imwrite(str(oracle/f'{case}_witness.png'),image)
            # Printed marker coordinates derive from the frozen texture generator.
            # Inspect its actual bounds separately; no decoder/reader result selects the case.
            dock_image,dock_meta=render([-2,-2,0],scene)
            cv2.imwrite(str(oracle/f'{case}_initial.png'),dock_image)
            initial=next(a for a in dock_meta['assets'] if a['actual_id']==7)
            old_image,old_meta=render([-1.5,-1.5,0],scene)
            cv2.imwrite(str(oracle/f'{case}_old_view.png'),old_image)
            old=next(a for a in old_meta['assets'] if a['actual_id']==7)
            line=blocked_segments(np.array([-2.,-2.,.8]),np.array([[witness[0],witness[1],.8]]),boxes)
            record.update(witness_pose=witness,known_map_path=path,witness_image_truth=target,
                initial_camera_truth=initial,old_camera_truth=old,witness_occluded_from_initial_lidar_origin=bool(line[0]))
            assert path is not None and target['usable_image'],record
        records.append(record)
    save(oracle/'feasibility.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Evaluator-only geometry and renderer checks; no controller or reader run',cases=records,
        assumptions='0.23m physical footprint; additional frozen0.52m routing inflation checked; ideal camera geometry; paths are oracle witnesses, not policy inputs'))
    manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=frozen['cases'],
        construction='fixed partitions and five assets cases per docs/connected_layout/protocol.md; no selection using mission/reader outcomes',
        files={str(p):sha(p) for p in templates.rglob('*') if p.is_file()},
        evaluator_files={str(p):sha(p) for p in oracle.rglob('*') if p.is_file()},
        pipeline_freeze_sha256=sha(root/'pipeline_freeze.json'),physical_outcomes_opened=False)
    save(root/'scenario_freeze.json',manifest)
    print(json.dumps([{k:v for k,v in r.items() if k in ['case','connected_floor_area_m2','all_footprint_free_space_connected','witness_pose','witness_occluded_from_initial_lidar_origin']} for r in records],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();build(a.root.resolve())
