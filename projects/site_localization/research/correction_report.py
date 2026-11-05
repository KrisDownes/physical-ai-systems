"""Mission-level reporting for the correction cycle; never a deployment input."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from transfer_report import mission


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--reserved',action='store_true')
    p.add_argument('--surface-comparison',action='store_true');a=p.parse_args();root=a.root
    if a.reserved:
        gate=json.loads((root/'collection_gate.json').read_text());assert gate['allowed_to_open']
        status=json.loads((root/'collection_status.json').read_text());assert status['state']=='collection_complete_outcomes_unopened'
        marker=root/'outcomes_opened.json'
        if not marker.exists():marker.write_text(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            note='Reserved cohort now development evidence for any subsequent changes.'),indent=2)+'\n')
    status_path=root/'collection_status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
    metadata={e['artifact_directory']:e['job'] for e in status.get('events',[]) if e.get('kind')=='process_health'}
    if a.surface_comparison:
        plan=json.loads((root/'plan.json').read_text())
        metadata.update({j['artifact_directory']:dict(role='full_deployment',repeat=j['repeat']) for j in plan['jobs']})
        records=[Path(j['artifact_directory'])/'run.json' for j in plan['jobs']]
    else:records=sorted(root.rglob('run.json'))
    rows=[]
    for path in records:
        folder=path.parent;run=json.loads(path.read_text());cfg=json.loads((folder/'cell.json').read_text())
        if run.get('physics_started') is False:
            rows.append(dict(run=folder.name,artifact_directory=str(folder),setup_invalid=True,wall_s=run['wall_s'],
                all_owned_processes_reaped=run['all_owned_processes_reaped']));continue
        row=mission(folder);row.update(layout=cfg['layout'],condition=cfg['condition'],physical_run=True)
        meta=metadata.get(str(folder.resolve()),{})
        if meta:row.update(method_role=meta['role'],repeat=meta['repeat'])
        else:row.update(method_role='full_acquisition' if cfg['mapping']=='new' else 'full_deployment',
                        repeat=int(folder.name.split('__repeat')[1][0]) if '__repeat' in folder.name else None)
        if row['method_role'].endswith('deployment'):
            if row['method_role']=='graph_deployment':
                decision=root/(folder.name.split('__setup_retry')[0]+'.preacceptance.json')
                row['acceptance_evidence']='Historical commissioning health, no current-site repair survey'
            else:
                source=Path(cfg['map_source']).parent.parent
                row['candidate_acquisition']=str(source)
                decision=source/('preacceptance_surface_policy.json' if a.surface_comparison else 'preacceptance.json')
                if (source/'run.json').exists():
                    acq=json.loads((source/'run.json').read_text());row['repair_wall_s']=acq['wall_s']
                    cost=source/'registration_cost.json'
                    if cost.exists():row['repair_wall_s']+=json.loads(cost.read_text())['wall_s']
                row['acceptance_evidence']='Current candidate acquisition and retained commissioning frame'
            row['preacceptance_record']=str(decision)
            row['decisions']=json.loads(decision.read_text()).get('rules',{}) if decision.exists() else {}
            row['operational_eligible']=row['decisions'].get('operational',{}).get('decision')=='provisionally_accept'
        rows.append(row)
    deployments=[r for r in rows if r.get('method_role','').endswith('deployment')]
    primary=[r for r in deployments if r['method_role']=='full_deployment']
    confusion={}
    for row in deployments:
        if 'mission_success' not in row:continue
        for name,rule in row.get('decisions',{}).items():
            c=confusion.setdefault(name,dict(accepted_success=0,accepted_failure=0,rejected_success=0,rejected_failure=0))
            key=('accepted_' if rule['decision']=='provisionally_accept' else 'rejected_')+('success' if row['mission_success'] else 'failure')
            c[key]+=1
    surveys=[r for r in rows if r.get('method_role')=='full_acquisition']
    development_gate=(len(primary)==4 and all(r.get('mission_success') and r.get('operational_eligible') for r in primary)
        and all((Path(r['artifact_directory'])/'controller_done.json').exists() for r in surveys))
    result=dict(split='reserved_at_collection_now_opened' if a.reserved else 'development',
        physical_executions=sum(r.get('physical_run',False) for r in rows),deployment_missions=len(deployments),
        distinct_layouts=sorted(set(r['layout'] for r in rows if r.get('physical_run'))),
        development_gate_passed=development_gate,primary_complete=sum(bool(r.get('mission_success')) for r in primary),
        primary_eligible_complete=sum(bool(r.get('mission_success') and r.get('operational_eligible')) for r in primary),
        primary_missions_observed=len(primary),primary_missions_planned=4,
        acceptance_confusion=confusion,missions=rows,
        limitations='Stops/timesteps correlated. Mapping runs are acquisition evidence, not independent deployment trials. Shared graphs/maps limit independence. Purposive simulated layouts; no population reliability or physical-transfer claim.')
    (root/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    fields=['run','layout','condition','method_role','repeat','mission_success','operational_eligible','physical_stops',
        'dock_after_completed_inspections','physical_dock_after_last_claim','max_claim_position_m','max_claim_heading_deg',
        'body_contacts','min_clearance_m','observed_sim_s','wall_s','repair_wall_s','cpu_s','largest_child_rss_kib','stop_reason',
        'all_owned_processes_reaped','setup_invalid']
    with (root/'missions.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='missions'},indent=2))


if __name__=='__main__':main()
