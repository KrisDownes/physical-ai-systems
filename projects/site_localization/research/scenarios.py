"""Development controls: frame metadata and actual geometry are separate.

The evaluator's mission-frame target poses stay fixed. The controller gets only
its declared map-frame transform and observations, never the new true geometry.
"""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def make(out, condition, remedy, severity=1.0):
    cfg=json.loads((ROOT/'sim/cell.json').read_text())
    tree=ET.parse(ROOT/'sim/cell.sdf');world=tree.getroot().find('world')
    cfg.update(condition=condition,scenario=condition,remedy=remedy,split='development',
               map_from_mission=[0,0,0],physical_target_frame='immutable commissioning mission frame',
               initialization='Known commissioned dock; no subsequent truth input',
               operator_events=[dict(type='known_dock_initialization',count=1,human_seconds=None)],
               reset_count=1,repeat_attempt_index=0,geometry_changed=False)
    original=list(cfg['obstacles'])
    if condition=='physical_change':
        # Storage/boards mask three upper-room boundary surfaces; targets and
        # direct waypoint routes remain traversable. No moved building walls.
        additions=[[-2.65,1.1,.12,3.3],[2.65,1.1,.12,3.3],[0,2.65,5.4,.12]]
        for i,(x,y,sx,sy) in enumerate(additions):
            m=ET.SubElement(world,'model',name=f'changed_storage_{i}')
            ET.SubElement(m,'static').text='true';ET.SubElement(m,'pose').text=f'{x} {y} .5 0 0 0'
            link=ET.SubElement(m,'link',name='link')
            for kind in ['collision','visual']:
                e=ET.SubElement(link,kind,name=kind);g=ET.SubElement(e,'geometry');b=ET.SubElement(g,'box')
                ET.SubElement(b,'size').text=f'{sx} {sy} 1'
        cfg['obstacles']=original+additions
        cfg['geometry_changed']=True;cfg['changes']=dict(type='added storage masking unchanged walls',boxes=additions)
    elif condition not in ['unchanged','frame_shift']:
        raise ValueError(condition)
    content=(ROOT/'sim/map.txt').read_text().splitlines()
    if condition=='frame_shift':
        header=content[0].split();header[3]=str(float(header[3])+.4);content[0]=' '.join(header)
        if remedy=='metadata_alignment':cfg['map_from_mission']=[.4,0,0]
        cfg['frame_change']=[.4,0,0]
        cfg['frame_change_information']='supplied map-origin edit metadata, not estimated simulator truth'
    cfg['commissioned_obstacles']=original
    (out/'map.txt').write_text('\n'.join(content)+'\n')
    (out/'cell.json').write_text(json.dumps(cfg,indent=2)+'\n')
    ET.indent(tree);tree.write(out/'cell.sdf',encoding='unicode')
    return cfg
