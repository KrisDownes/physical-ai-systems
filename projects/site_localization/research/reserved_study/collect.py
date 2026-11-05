"""Frozen reserved collection; never reads judge traces or physical outcomes."""
import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from scipy.spatial import cKDTree

F=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(F/'research'))
from correction_pipeline import anchor,surface_audit
from correction_acceptance import analyze
from transfer_acceptance import analyze_alignment
from frames import compose,inverse
from register_maps import occupied_points,transform,summary,conditioning
from types import SimpleNamespace
import reserved_layout


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify():
    for n,h in load(F/'manifest.json')['files'].items():
        p=Path(n[9:]) if n.startswith('external:') else F/n
        assert sha(p)==h,n


def main(root):
    verify();root.mkdir(parents=True,exist_ok=False)
    manifest=load(F/'manifest.json');protocol=load(F/'protocol.json')
    events=[];lock=threading.Lock();serial=0
    def save_event(e):
        with lock:
            e=dict(recorded_utc=now(),**e);events.append(e)
            write(root/'events.json',events)
    def job(condition,role,repeat,seed,extra,selector=None,selection=None):
        nonlocal serial
        serial+=1
        return dict(condition=condition,role=role,repeat=repeat,seed=seed,domain=190+serial,
            selector=selector,selection=selection,name=f'service_island__{condition}__{role}__repeat{repeat}'+('__'+selector if selector else ''),extra=list(map(str,extra)))
    def run(j):
        verify()
        for attempt in range(2):
            out=root/(j['name']+('__prephysics_retry1' if attempt else ''))
            cmd=['/usr/bin/python3',str(F/'study/reserved_trial.py'),'--fixture-condition',j['condition'],'--out',str(out),
                 '--seed',str(j['seed']),'--domain',str(j['domain']),'--wall-limit','620',*j['extra']]
            e=dict(kind='attempt_started',job=j,attempt=attempt,output=str(out),command=cmd,reset_requested=1)
            save_event(e);start=time.monotonic()
            log=root/(out.name+'.log')
            with log.open('w') as stream:
                p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=p.wait(timeout=750)
                except subprocess.TimeoutExpired:
                    p.send_signal(signal.SIGINT)
                    try:p.wait(timeout=60)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    code='outer_timeout'
            health=load(out/'run.json') if (out/'run.json').exists() else {}
            lock_refusal=not out.exists() and 'is already in use' in log.read_text()
            invalid=health.get('physics_started') is False or lock_refusal
            record=dict(kind='attempt_finished',job=j,attempt=attempt,output=str(out),returncode=code,wall_s=time.monotonic()-start,
                physics_started=health.get('physics_started'),stop_reason=health.get('stop_reason'),score_returncode=health.get('score_returncode'),
                all_owned_processes_reaped=health.get('all_owned_processes_reaped'),reset_count=health.get('reset_count',0),prephysics_invalid=invalid)
            if out.exists():
                assert sha(out/'cell.sdf')==sha(F/'fixtures'/j['condition']/'cell.sdf')
                if j['selection']:(out/'calibration_selection.json').write_bytes(Path(j['selection']).read_bytes())
            save_event(record);print(json.dumps({k:record[k] for k in ['kind','output','stop_reason','prephysics_invalid']}),flush=True)
            if attempt==0 and invalid:continue
            return out
    def stage(jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            return list(pool.map(run,jobs))
    dock=reserved_layout.spec()['start']
    def calibrate(path,anchor_path,role):
        try:
            surface_audit(path,dock)
            raw=load(path/'surface_alignment.json');source,_=occupied_points(path/'source_surface.txt');target,_=occupied_points(path/'target_surface.txt');tree=cKDTree(target)
            records={}
            for selector,key in [('capped','objective_m2'),('trimmed','trimmed_objective_m2')]:
                i=min(range(len(raw['candidates'])),key=lambda i:raw['candidates'][i][key]);c=raw['candidates'][i]
                a=copy.deepcopy(raw);a.update(target_from_source=c['pose'],selected=c)
                a['parameters']['selection_objective']=selector
                moved=transform(source,np.array(c['pose']));d,ids=tree.query(moved);rev,_=cKDTree(moved).query(target)
                a['source_to_target']=summary(d,.12);a['target_to_source']=summary(rev,.12)
                a['conditioning']=conditioning(moved,target,tree,d,ids,SimpleNamespace(inlier_distance=.12))
                from register_maps import rotation
                matrix=np.eye(3);matrix[:2,:2]=rotation(c['pose'][2]);matrix[:2,2]=c['pose'][:2];a['target_from_source_matrix']=matrix.tolist()
                fit=path/(selector+'_alignment.json');write(fit,a)
                decision=analyze(fit,path/'source_surface.txt',path/'target_surface.txt',path,c['pose'],anchor_path,frame_policy='surface_fit')
                strict=analyze_alignment(fit,path/'source_surface.txt',path/'target_surface.txt')
                record=dict(selector=selector,index=i,target_from_source=c['pose'],map_from_mission=inverse(c['pose']).tolist(),
                    selected=c,original_candidates=raw['candidates'],shared_alignment_sha256=sha(path/'surface_alignment.json'),
                    map_sha256=sha(path/'mapper/map.txt'),decision_recorded_utc=now(),acceptance=decision,existing_strict_screens=strict,
                    audit_only_if_rejected=True,role=role)
                write(path/(selector+'_selection.json'),record);records[selector]=record
            cfg=load(path/'controller_config.json');poses=np.array([cfg['start']]+[r['pose'] for r in cfg['route']])
            x=records['capped']['target_from_source'];y=records['trimmed']['target_from_source'];q=compose(inverse(x),poses)
            maxd=float(np.linalg.norm(compose(y,q)[:,:2]-poses[:,:2],axis=1).max());yaw=float(abs(np.arctan2(np.sin(x[2]-y[2]),np.cos(x[2]-y[2]))))
            # Shared execution only for numerically identical effective poses, not merely small differences.
            shared=bool(np.allclose(x,y,rtol=0,atol=1e-9))
            group=dict(acquisition=str(path),anchor_survey=str(anchor_path),records=records,shared_execution=shared,
                       max_route_disagreement_m=maxd,yaw_disagreement_rad=yaw,role=role)
            write(path/'comparison_before_outcomes.json',group);return group
        except (ValueError,RuntimeError,OSError,StopIteration,subprocess.TimeoutExpired) as error:
            save_event(dict(kind='calibration_dependency_failure',path=str(path),role=role,detail=str(error)))
            return None
    commission=job('unchanged','commission',0,5900,['--mapping','new','--localizer','slam_toolbox','--map-from-mission',*inverse(dock)])
    commission_path=run(commission);commission_group=None
    try:
        anchor(commission_path,dock);commission_group=calibrate(commission_path,commission_path,'commission')
    except (OSError,ValueError,StopIteration) as error:save_event(dict(kind='commission_dependency_failure',detail=str(error)))
    graph=commission_path/'mapper/session';graph_hashes={str(p):sha(p) for p in [graph.with_suffix('.posegraph'),graph.with_suffix('.data')] if p.exists()}
    graph_ready=commission_group is not None and len(graph_hashes)==2
    write(root/'commission_graph_hashes.json',graph_hashes)
    acquisitions=[]
    for ci,condition in enumerate(['unchanged','rearrangement']):
        for repeat in range(2):
            acquisitions.append(job(condition,'full_acquisition',repeat,6000+ci*100+repeat,
                ['--mapping','new','--localizer','slam_toolbox','--map-from-mission',*inverse(dock)]))
        if graph_ready:
            acquisitions.append(job(condition,'continued_acquisition',0,6000+ci*100,
                ['--mapping','continued','--localizer','slam_toolbox','--session',graph,
                 '--map-from-mission',*commission_group['records']['capped']['map_from_mission']]))
        else:save_event(dict(kind='missing_graph_dependency',condition=condition))
    paths=stage(acquisitions);groups=[]
    for j,path in zip(acquisitions,paths):
        if not (path/'mapper/map.txt').exists():save_event(dict(kind='missing_candidate_map',job=j,path=str(path)));continue
        try:
            if j['role']=='full_acquisition':anchor(path,dock);anchor_path=path
            else:
                anchor_path=commission_path;write(path/'frame_lineage.json',dict(loaded_graph_hashes=graph_hashes,anchor_survey=str(anchor_path)))
            g=calibrate(path,anchor_path,j['role'].replace('acquisition','deployment'))
            if g:g.update(condition=j['condition'],repeat=j['repeat'],seed=j['seed']);groups.append(g)
        except (OSError,ValueError,StopIteration) as error:save_event(dict(kind='missing_frame_dependency',job=j,detail=str(error)))
    if graph_ready:
        for ci,condition in enumerate(['unchanged','rearrangement']):
            g=copy.deepcopy(commission_group);g.update(condition=condition,repeat=0,seed=6000+ci*100,role='graph_deployment');groups.append(g)
    write(root/'selections_before_outcomes.json',dict(recorded_utc=now(),groups=groups,planned_full_maps=4,physical_outcomes_inspected=False))
    pairs=[]
    for g in groups:
        source_path=Path(g['acquisition']);pair=[]
        for selector in ['capped'] if g['shared_execution'] else ['capped','trimmed']:
            record=g['records'][selector]
            extra=(['--mapping','localization','--localizer','slam_toolbox','--session',graph] if g['role']=='graph_deployment'
                   else ['--remedy','acquired_map','--map',source_path/'mapper/map.txt'])
            extra+=['--map-from-mission',*record['map_from_mission']]
            j=job(g['condition'],g['role'],g['repeat'],g['seed'],extra,selector,str(source_path/(selector+'_selection.json')))
            j['selectors_evaluated']=['capped','trimmed'] if g['shared_execution'] else [selector]
            j['operational_decisions']={s:g['records'][s]['acceptance']['rules']['operational'] for s in j['selectors_evaluated']}
            pair.append(j)
        pairs.append(pair)
    write(root/'deployment_plan.json',dict(recorded_utc=now(),pairs=pairs))
    def execute_pair(pair):return [run(j) for j in pair]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(execute_pair,pairs))
    for n,h in graph_hashes.items():assert sha(n)==h,'Shared graph modified'
    verify();write(root/'collection_complete.json',dict(finished_utc=now(),outcomes_unopened=True,revision=manifest['revision'],
        planned_full_maps=4,realized_groups=len(groups),unique_deployment_jobs=sum(map(len,pairs)),planned_base_runs=15,
        note='Missing dependencies remain workflow failures, not excluded cases.'))
    print('Collection complete; physical outcomes remain unopened.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out.resolve())
