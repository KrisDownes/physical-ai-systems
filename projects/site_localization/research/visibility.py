"""Evaluator-only analytic ray check for physical-change visibility.

This reads simulator construction geometry. It must never be a method input.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def ranges(pose,boxes,count=181,cap=8):
    angles=pose[2]+np.linspace(-np.pi,np.pi,count);directions=np.c_[np.cos(angles),np.sin(angles)]
    result=np.full(count,float(cap));origin=np.asarray(pose[:2])
    for x,y,sx,sy in boxes:
        lower=np.array([x-sx/2,y-sy/2]);upper=np.array([x+sx/2,y+sy/2])
        near=np.full((count,2),-np.inf);far=np.full((count,2),np.inf);valid=np.ones(count,dtype=bool)
        for axis in range(2):
            moving=abs(directions[:,axis])>1e-12
            a=(lower[axis]-origin[axis])/directions[moving,axis]
            b=(upper[axis]-origin[axis])/directions[moving,axis]
            near[moving,axis]=np.minimum(a,b);far[moving,axis]=np.maximum(a,b)
            valid[~moving]=(lower[axis]<=origin[axis]<=upper[axis])
        entry=near.max(axis=1);exit=far.min(axis=1);valid &= exit>=np.maximum(entry,0)
        hit=np.where(entry>=0,entry,exit)
        result=np.minimum(result,np.where(valid,hit,cap))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());rows=[]
    for i,target in enumerate(cfg['goals']):
        old=ranges(target,cfg['commissioned_obstacles']);new=ranges(target,cfg['obstacles'])
        rows.append(dict(goal=i,changed_beams=int(np.sum(abs(new-old)>.05)),total_beams=len(old),
                         fraction_changed=float(np.mean(abs(new-old)>.05))))
    result=dict(scope='evaluation-only ideal rays from simulator geometry; not online observations',targets=rows)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
