"""Evaluator-only fresh central-doorway cases; no method inputs from truth."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/connected_layout'))
from prepare import floor_grid
sys.path.insert(0,str(ROOT/'research/moved_asset'))
from planning import GridPlanner
from render import render

def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'results/connected_layout_20261025/templates'
    boxes=[[-3.2,0,.15,6.55,1.],[3.2,0,.15,6.55,1.],
           [0,-3.2,6.55,.15,1.],[0,3.2,6.55,.15,1.],
           [.5,-2.05,.18,2.3,1.],[.5,2.05,.18,2.3,1.],
           [-2.35,.5,1.7,.18,1.]]
    oracle=out/'evaluator_preflight';oracle.mkdir();templates=out/'templates';templates.mkdir()
    axis,points,free,connected,clearance=floor_grid(boxes)
    truthmap=oracle/'truth_grid.txt'
    with truthmap.open('w') as f:
        f.write(f'{len(axis)} {len(axis)} .05 {axis[0]} {axis[0]}\n')
        np.savetxt(f,np.where((clearance==0).reshape(free.shape),1,-1),fmt='%d')
    planner=GridPlanner(truthmap,.52)
    assert bool(np.all(connected[free])),'disconnected footprint-free floor'
    records=[]
    for case,source,target in [('fresh_visible','visible',[0,-1.5,0]),
                               ('fresh_moved_wrong','wrong_asset',[2.35,-1.4,0])]:
        dst=templates/case;shutil.copytree(base/source,dst)
        scene=load(dst/'private/scene.json');scene['obstacles']=boxes
        for asset in scene['assets']:
            if asset['id']==7:asset['pose']=target
        save(dst/'private/scene.json',scene)
        evaluation=load(dst/'private/evaluation.json');evaluation.update(case=case,physical_obstacles=[b[:4] for b in boxes])
        save(dst/'private/evaluation.json',evaluation)
        tree=ET.parse(dst/'private/world.sdf');world=tree.getroot().find('world')
        for i,(x,y,sx,sy,h) in enumerate(boxes):
            model=world.find(f"model[@name='partition_{i}']")
            assert model is not None,(case,i)
            model.find('pose').text=f'{x} {y} {h/2} 0 0 0'
            for size in model.findall('.//box/size'):size.text=f'{sx} {sy} {h}'
        panel=world.find("model[@name='asset_panel_7']")
        assert panel is not None
        panel.find('pose').text=f'{target[0]} {target[1]} .8 0 0 {target[2]}'
        tree.write(dst/'private/world.sdf',encoding='unicode')
        witness=[target[0]-1.1*math.cos(target[2]),target[1]-1.1*math.sin(target[2]),target[2]]
        path=planner.path([-2,-2],witness)
        image,metadata=render(witness,scene)
        cv2.imwrite(str(oracle/f'{case}_witness.png'),image)
        observed=next(a for a in metadata['assets'] if a['actual_id']==7)
        assert path is not None and observed['usable_image'],(case,path,observed)
        records.append(dict(case=case,connected_floor_cells=int(connected.sum()),
                            evaluator_only_target=target,evaluator_only_witness=witness,
                            evaluator_only_known_map_path=path,witness_image_truth=observed))
    save(oracle/'feasibility.json',dict(cases=records,scope='evaluator-only connected floor and render witness; no reader/policy scoring'))
    config=ROOT/'research/constraint_repair/slam_params_radius_1p5.yaml'
    sourcefiles=[config,ROOT/'research/continuous_asset/run.py',ROOT/'docs/constraint_repair/protocol.md']
    save(out/'freeze.json',dict(cases=['fresh_visible','fresh_moved_wrong'],order=['fresh_visible','fresh_moved_wrong'],
         seed=7403,wall_limit_s=800,config_sha256=sha(config),source_hashes={str(p):sha(p) for p in sourcefiles},
         template_hashes={str(p):sha(p) for p in templates.rglob('*') if p.is_file()},
         evaluator_hashes={str(p):sha(p) for p in oracle.rglob('*') if p.is_file()},
         physical_contract='continuous-v2 unchanged; 650s/35m whole mission; no dock return; no reset',
         correction='only loop_search_maximum_distance: 3.0 -> 1.5; do_loop_closing true',
         outcome_scored=False))
    print(json.dumps(records,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('out',type=Path);args=a.parse_args();main(args.out.resolve())
