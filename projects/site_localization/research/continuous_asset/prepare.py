"""Private scenario construction; no changed geometry is an input to robot code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    for cohort in ['development','fresh']:
        for case in ['unchanged','moved','wrong_asset','absent']:
            dst=out/cohort/case;shutil.copytree(ROOT/'results/autonomous_asset_20261024/v2/templates'/case,dst)
            # The inherited supplied grid is setup baggage: remove it from the
            # new template so it cannot accidentally reach either controller.
            (dst/'map.txt').rename(dst/'private/unused_original_supplied_map.txt')
            evaluation=load(dst/'private/evaluation.json');evaluation.update(max_sim_s=650,max_path_m=35)
            if cohort=='fresh':
                scene=load(dst/'private/scene.json');scene['obstacles'][4][:2]=[.2,.45];scene['obstacles'][5][:2]=[-2.3,.7]
                if case!='unchanged':scene['obstacles'][7]=[.2,-.6,.6,.6,1.]
                for asset in scene['assets']:
                    if asset['id']==7 and case!='unchanged':asset['pose']=[1.9,1.5,0.]
                evaluation['physical_obstacles']=[o[:4] for o in scene['obstacles']]
                tree=ET.parse(dst/'private/world.sdf');world=tree.getroot().find('world')
                for model in world.findall('model'):
                    name=model.get('name')
                    index=int(name.split('_')[1]) if name in ['obstacle_4','obstacle_5'] else 7 if name=='relocated_cabinet' else None
                    if index is not None:
                        x,y,sx,sy,_=scene['obstacles'][index];model.find('pose').text=f'{x} {y} .5 0 0 0'
                        for size in model.findall('.//box/size'):size.text=f'{sx} {sy} 1'
                    if name=='asset_panel_7' and case!='unchanged':model.find('pose').text='1.9 1.5 .8 0 0 0'
                tree.write(dst/'private/world.sdf',encoding='unicode');save(dst/'private/scene.json',scene)
            save(dst/'private/evaluation.json',evaluation)
    save(out/'scenario_freeze.json',dict(files={str(p):sha(p) for p in out.rglob('*') if p.is_file()},fresh_outcomes_opened=False))
    print('Prepared and frozen scenarios',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();main(a.out.resolve())
