"""Read-only causal replay of recorded acquisition failures; no method changes."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/moved_asset'))
from planning import GridPlanner

def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main(root,out):
    out.mkdir(parents=True,exist_ok=False)
    p=root/'acquisition__unchanged__exploration';events=rows(p/'exploration.jsonl');pose=events[-1]['pose']
    records=[]
    for f in [*sorted((p/'decision_maps').glob('*.txt'))[-2:],p/'live_planning_grid.txt']:
        g=GridPlanner(f);u,v=np.rint((np.asarray(pose[:2])-[g.ox,g.oy])/g.res).astype(int)
        with f.open() as s:s.readline();a=np.loadtxt(s,dtype=int)
        records.append(dict(map=str(f),pose=pose,cell=[int(v),int(u)],native_state=int(a[v,u]),
            distance_to_nonfree_m=float(g.distance[v,u]),required_m=.45+g.res/np.sqrt(2),start_usable=g.index(pose) is not None))
    save(out/'unchanged_start_rejection.json',dict(records=records,interpretation='The same terminal pose is outside the inflated free set even in previous decision grids. This contradicts a map-update-only explanation. The path follower entered a neighboring excluded cell; map refresh exposed it. No physical collision is implied by the hard planner margin.'))
    p=root/'acquisition__moved__exploration';events=rows(p/'exploration.jsonl');wheel=np.genfromtxt(p/'mapper/wheel_odometry.csv',delimiter=',',names=True)
    distance=np.r_[0,np.cumsum(np.hypot(np.diff(wheel['x']),np.diff(wheel['y'])))];paths=[]
    for e in events:
        if e['type']!='path':continue
        travelled=float(np.interp(e['time'],wheel['timestamp'],distance));length=float(sum(np.linalg.norm(np.asarray(a)-b) for a,b in zip(e['points'],e['points'][1:])))
        decision=next((d for d in reversed(events) if d['type']=='frontier_decision' and d['time']<=e['time']),None)
        return_length=None
        if decision:
            g=GridPlanner(Path(decision['map_file']));home=g.path(e['goal'],[0,0])
            if home:return_length=home['length_m']
        paths.append(dict(time=e['time'],phase=e['phase'],distance_already_m=travelled,planned_leg_m=length,
            planned_return_from_goal_m=return_length,remaining_total_distance_m=20-travelled,
            predicted_leg_plus_return_fits=None if return_length is None else length+return_length<=20-travelled))
    crossing=int(np.searchsorted(distance,15))
    save(out/'moved_return_budget.json',dict(paths=paths,first_15m_crossing_s=float(wheel['timestamp'][crossing]),
        interpretation='Exploration distance reserve is checked only when selecting a new leg. A long committed leg consumes the reserve before return begins. Existing map path costs could have detected this without truth. No budget or policy is changed in this replay.'))

if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
