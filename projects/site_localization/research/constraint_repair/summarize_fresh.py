"""Independent evaluator summary for fixed fresh closed-loop cases."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import truth_pose

def load(p):return json.loads(p.read_text())
def pose_at(t,truth):return np.array([np.interp(t,truth[:,0],truth[:,i]) for i in (1,2,3)])

def main():
    base=ROOT/'results/constraint_repair_20261026/fresh';rows=[]
    for case,dirname in [('fresh_visible','mission_fresh_visible_rerun01'),('fresh_moved_wrong','mission_fresh_moved_wrong')]:
        m=base/dirname
        if not (m/'outcome.json').exists():continue
        x=load(m/'outcome.json');run=load(m/'run.json');truth=truth_pose(m)
        graph=base/'analysis'/f'{case}_graph'
        map_eval=None
        if Path(str(graph)+'_nodes.csv').exists():
            with Path(str(graph)+'_nodes.csv').open() as f:nodes={int(r['id']):{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)}
            errors=[float(np.linalg.norm(np.array([n['x']-2,n['y']-2])-pose_at(n['time'],truth)[:2])) for n in nodes.values()]
            bad=0;legit=0
            with Path(str(graph)+'_edges.csv').open() as f:
                for e in csv.DictReader(f):
                    u,v=int(e['source']),int(e['target'])
                    if v-u<10:continue
                    a,b=pose_at(nodes[u]['time'],truth),pose_at(nodes[v]['time'],truth)
                    d=b[:2]-a[:2];c,s=math.cos(a[2]),math.sin(a[2]);z=(c*d[0]+s*d[1],-s*d[0]+c*d[1])
                    if math.dist(z,(float(e['dx']),float(e['dy'])))>.5:bad+=1
                    else:legit+=1
            map_eval=dict(nodes=len(nodes),endpoint_position_error_m=round(errors[-1],4),max_position_error_m=round(max(errors),4),incorrect_cross_time_links=bad,legitimate_cross_time_links=legit)
        final=next((o for o in reversed(x.get('all_observations',[])) if o.get('valid_physical_inspection')),None)
        rows.append(dict(case=case,attempt=dirname,complete_contract=x['complete_contract'],success_claim=x['success_claim'],false_completion=x['false_completion'],wrong_asset_acceptances=x['wrong_asset_acceptances'],
                         termination=x['termination'],inspection_reason=x['inspection_reason'],contacts=x['contacts'],clearance_m=x['clearance_m'],
                         total_sim_s=x['total_sim_s'],total_path_m=x['total_path_m'],acquisition_time_s=x['acquisition_time_s'],
                         acquisition_distance_m=x['acquisition_physical_distance_m'],exploration_termination=x['acquisition']['exploration_termination'],
                         handoff_success=x['handoff_success'],handoff_seed_error_m=x['handoff_seed_position_error_m'],
                         observations=x['observations'],recoveries=x['recoveries'],final_physical_observation=final,
                         map_eval=map_eval,resets=run['resets_after_start'],interventions=x['interventions'],owned_processes_reaped=run['all_owned_processes_reaped']))
    (base/'analysis/missions.json').write_text(json.dumps(rows,indent=2)+'\n')
    for r in rows:print(r['case'],r['complete_contract'],r['termination'],r['total_sim_s'],r['total_path_m'],r['map_eval'])

if __name__=='__main__':main()
