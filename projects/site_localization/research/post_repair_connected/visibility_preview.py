"""Map-only coverage preview; never reads scenario or evaluator truth."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/moved_asset'))
sys.path.insert(0,str(ROOT/'research/continuous_asset'))
from buffered_planning import BufferedPlanner

RUN=ROOT/'results/post_repair_connected_20261027/mission_wrong_asset/inspection'
OUT=ROOT/'results/post_repair_connected_20261027/search_visibility_preview_v2.json'

def main():
    config=json.loads((RUN/'public/policy.json').read_text())
    planner=BufferedPlanner(RUN/'map.txt')
    with (RUN/'map.txt').open() as f:
        f.readline();grid=np.loadtxt(f,dtype=int)
    yy,xx=np.nonzero(grid[::2,::2]==-1)
    cells=np.c_[xx*2,yy*2]
    points=np.c_[planner.ox+cells[:,0]*planner.res,planner.oy+cells[:,1]*planner.res]
    def visible(station):
        station=np.asarray(station[:2],float)
        distance=np.linalg.norm(points-station,axis=1)
        keep=(distance>=.6)&(distance<=5)
        idx=np.flatnonzero(keep)
        if not len(idx):return np.zeros(len(points),bool)
        # Fixed 2.5 cm ray samples, at least two per occupancy cell.
        ray=station[None,None,:]+np.linspace(0,1,201)[None,:,None]*(points[idx,None,:]-station[None,None,:])
        ix=np.rint((ray[:,:,0]-planner.ox)/planner.res).astype(int)
        iy=np.rint((ray[:,:,1]-planner.oy)/planner.res).astype(int)
        inside=(ix>=0)&(ix<planner.w)&(iy>=0)&(iy<planner.h)
        ix=np.clip(ix,0,planner.w-1);iy=np.clip(iy,0,planner.h-1)
        result=np.zeros(len(points),bool)
        result[idx]=np.all(inside&(grid[iy,ix]==-1),axis=1)
        return result
    options=[]
    for x in config['search_coordinates']:
        for y in config['search_coordinates']:
            p=np.array([x,y]);
            if planner.index(p) is not None:options.append(p)
    events=[json.loads(s) for s in (RUN/'method.jsonl').read_text().splitlines()]
    states=[]
    for label,visited,actual_goal in [
        ('after_old',[[-1.5,-1.5]],[0,1.5]),
        ('after_old_and_central',[[-1.5,-1.5],[0,1.5]],[-1.5,1.5])]:
        action=next(e for e in events if e['event']=='plan' and e['purpose']=='search' and
                    np.linalg.norm(np.asarray(e['goal'][:2])-actual_goal)<.01 and
                    np.linalg.norm(np.asarray(e['start'][:2])-np.asarray(visited[-1]))<.15)
        start=action['start'][:2]
        seen=np.zeros(len(points),bool)
        for p in visited:seen|=visible(p)
        ranked=[]
        for p in options:
            if any(np.linalg.norm(p-q)<.5 for q in visited):continue
            route=planner.path(start,p)
            if route is None:continue
            mask=visible(p)
            gain=int((mask&~seen).sum())*.01 # sampled 2x2 cells at 5 cm resolution
            length=float(route['length_m'])
            ranked.append(dict(station=p.tolist(),path_m=length,weighted_graph_cost=float(route['graph_cost']),new_visible_area_m2=gain,
                               gain_per_effort=gain/(length+1.)))
        states.append(dict(state=label,visited=visited,start=start,actual_next=action['goal'][:2],
                           eligible=ranked,
                           nearest=min(ranked,key=lambda r:(r['weighted_graph_cost'],*r['station']))['station'],
                           max_gain=max(ranked,key=lambda r:r['new_visible_area_m2'])['station'],
                           max_gain_per_effort=max(ranked,key=lambda r:r['gain_per_effort'])['station']))
    result=dict(protocol='docs/post_repair_connected/search_preview.md',
                inputs=['inspection/map.txt','inspection/public/policy.json','inspection/method.jsonl decision poses'],
                truth_input=False,map_free_sampled_points=len(points),sample_area_m2=.01,
                radius_m=[.6,5.],states=states,
                limit='2D mapped-free raycast is only a visibility upper bound; no object orientation, elevated occlusion or detection model')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    for s in states:
        print(s['state'],'actual',s['actual_next'],'max_gain',s['max_gain'],
              'max_gain_per_effort',s['max_gain_per_effort'])

if __name__=='__main__':main()
