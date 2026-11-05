"""Open the completed cohort once; numerical physical judges are unchanged."""
import argparse
import datetime
import json
from pathlib import Path
import sys
F=Path(__file__).resolve().parents[1];sys.path.insert(0,str(F/'research'))
from transfer_report import mission,trajectory
import numpy as np
from frames import compose,inverse


def load(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main(root):
    assert load(root/'collection_complete.json')['outcomes_unopened']
    marker=root/'outcomes_opened.json'
    if not marker.exists():write(marker,dict(opened_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),subsequent_changes_require_fresh_validation=True))
    events=load(root/'events.json');rows=[]
    for e in events:
        if e['kind']!='attempt_finished':continue
        p=Path(e['output']);row=dict(**e)
        if (p/'outcome.json').exists():
            m=mission(p);row.update(m)
            row['layout']='service_island';row['condition']=e['job']['condition'];row['role']=e['job']['role']
            row['physical_mission_completion_s']=load(p/'outcome.json').get('dock_completion_time_s')
            try:
                cfg=load(p/'controller_config.json');t,xy,yaw=trajectory(p)
                ev=[x['event'] for x in map(json.loads,(p/'judge.jsonl').read_text().splitlines()) if x['type']=='event' and x['event']['type']=='claimed_stop']
                component=[]
                for event in ev:
                    i=event['goal'];stamp=event['time'];estimate=compose(inverse(cfg['map_from_mission']),event['estimate'])
                    physical=np.array([np.interp(stamp,t,xy[:,0]),np.interp(stamp,t,xy[:,1])])
                    component.append(dict(goal=i,claim_time_s=stamp,controller_residual_site_m=(estimate[:2]-np.array(cfg['goals'][i][:2])).tolist(),physical_minus_estimated_site_m=(physical-estimate[:2]).tolist()))
                row['claim_error_decomposition']=component
            except (AssertionError,ValueError,OSError,KeyError,IndexError) as error:row['decomposition_unavailable']=str(error)
        else:row['unjudged']=True
        rows.append(row)
    physical={r['output']:r for r in rows}
    pairs=[]
    for pair in load(root/'deployment_plan.json')['pairs']:
        byselector={}
        for j in pair:
            candidates=[r for r in rows if r['job']['name']==j['name'] and not r.get('prephysics_invalid')]
            r=candidates[-1] if candidates else None
            for s in j['selectors_evaluated']:
                eligible=j['operational_decisions'][s]['decision']=='provisionally_accept'
                byselector[s]=dict(physical_execution=None if r is None else r['output'],eligible=eligible,
                    operational_decision=j['operational_decisions'][s],physical_success=bool(r and r.get('mission_success')),
                    accepted_complete=bool(eligible and r and r.get('mission_success')),
                    max_claim_position_m=None if r is None else r.get('max_claim_position_m'),
                    dock=None if r is None else r.get('dock_after_completed_inspections'))
        j=pair[0];pairs.append(dict(condition=j['condition'],role=j['role'],repeat=j['repeat'],seed=j['seed'],selectors=byselector,shared_execution=len(pair)==1))
    full=[p for p in pairs if p['role']=='full_deployment']
    summary={}
    for s in ['capped','trimmed']:
        summary[s]=dict(planned_full_workflows=4,observed_full_workflows=len(full),
            eligible=sum(p['selectors'][s]['eligible'] for p in full),
            physical_success=sum(p['selectors'][s]['physical_success'] for p in full),
            accepted_complete=sum(p['selectors'][s]['accepted_complete'] for p in full),
            bounded_all_four_criterion_met=len(full)==4 and all(p['selectors'][s]['accepted_complete'] for p in full),
            deployment_reliability_qualified=False)
    result=dict(scope='One purposive new environment, unchanged/rearrangement, two independent full acquisitions per condition; no population reliability inference.',
        primary=summary,pairs=pairs,attempts=rows,distinct_environments=1,
        unique_physics_executions=sum(bool(r.get('physics_started')) for r in rows),
        unique_acquisition_executions=sum(bool(r.get('physics_started')) and r['job']['role'].endswith(('acquisition','commission')) for r in rows),
        unique_deployment_executions=sum(bool(r.get('physics_started')) and r['job']['role'].endswith('deployment') for r in rows),
        uncertainty='Finite cohort counts and within-map paired comparisons only. One environmental cluster cannot estimate cross-site variance; visits/seeds/shared executions are not independent sites.',
        operational_rule='Existing survey/frame/convergence/surface health only. Strict residual/support/ambiguity comparators remain separate in pre-outcome selection records.')
    write(root/'result.json',result);print(json.dumps(dict(primary=summary,pairs=pairs,counts={k:v for k,v in result.items() if k.startswith('unique_')}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
