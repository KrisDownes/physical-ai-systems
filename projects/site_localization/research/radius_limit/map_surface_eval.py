"""Evaluator-only lidar endpoint-to-physical-wall diagnostic for recorded replays."""
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/radius_limit_20261026'
SOURCES={
    'fresh_moved_wrong':ROOT/'results/constraint_repair_20261026/fresh/mission_fresh_moved_wrong',
    'occluded':ROOT/'results/connected_layout_20261025/mission_occluded',
    'regression_visible':ROOT/'results/connected_layout_20261025/mission_visible',
}

def frames(path):
    found={}
    with path.open() as f:
        next(f)
        for line in f:
            q=np.fromstring(line,sep=' ')
            if len(q)<6:continue
            n=int(q[4]);assert len(q)==5+2*n
            found[round(float(q[0]),5)]=(q[5::2],q[6::2])
    return found

def distances(points,boxes):
    best=np.full(len(points),np.inf)
    for x,y,sx,sy,*_ in boxes:
        dx=np.abs(points[:,0]-x)-sx/2;dy=np.abs(points[:,1]-y)-sy/2
        outside=np.hypot(np.maximum(dx,0),np.maximum(dy,0))
        inside=np.minimum(-dx,-dy)
        best=np.minimum(best,np.where((dx>0)|(dy>0),outside,inside))
    return best

def main():
    result=[]
    for case,mission in SOURCES.items():
        scene=json.loads((mission/'private/scene.json').read_text())
        boxes=scene['obstacles'];recorded=frames(mission/'acquisition/mapper/sensor_frames.txt')
        for tag,radius in [('3m',3.0),('1p5m',1.5)]:
            stem=f'{case}_{tag}'+('_v2' if case=='fresh_moved_wrong' else '')
            values=[];count=0
            for line in (OUT/f'{stem}.log').read_text().splitlines():
                if not line.startswith('POSE,'):continue
                _,id,t,x,y,yaw=line.split(',');key=round(float(t),5)
                assert key in recorded,(case,tag,id,key)
                ranges,angles=recorded[key];ok=np.isfinite(ranges)&(ranges>.05)&(ranges<7.9)
                a=angles[ok]+float(yaw);r=ranges[ok]
                endpoints=np.column_stack((float(x)-2+r*np.cos(a),float(y)-2+r*np.sin(a)))
                values.extend(distances(endpoints,boxes));count+=1
            d=np.asarray(values)
            result.append(dict(case=case,radius_m=radius,processed_scans=count,finite_returns=len(d),
                               median_distance_to_true_wall_m=round(float(np.median(d)),4),
                               p90_distance_to_true_wall_m=round(float(np.quantile(d,.9)),4),
                               fraction_within_0_1m=round(float(np.mean(d<=.1)),4),
                               note='Evaluator-only 2D box-surface diagnostic; includes correlated rays and ignores free-space topology'))
    (OUT/'map_surface_result.json').write_text(json.dumps(result,indent=2)+'\n')
    for q in result:print(q)

if __name__=='__main__':main()
