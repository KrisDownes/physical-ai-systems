"""Single focused reserved collection; physical outcomes remain unopened."""
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def verify(freeze):
    record=json.loads((freeze/'manifest.json').read_text())
    for name,digest in record['files'].items():
        path=Path(name[9:]) if name.startswith('external:') else freeze/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen method changed: '+name)
    study=json.loads((freeze/'study_manifest.json').read_text())
    for name,digest in study['files'].items():
        if hashlib.sha256((freeze/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen study changed: '+name)
    return record['revision']


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();freeze=a.freeze.resolve();root=a.out.resolve();revision=verify(freeze)
    sys.path.insert(0,str(freeze/'research'))
    from correction_pipeline import anchor,surface_audit
    from correction_acceptance import analyze
    from frames import inverse
    spec=importlib.util.spec_from_file_location('reserved_layout',freeze/'study/reserved_layout.py')
    geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    dock=geometry.spec()['start'];root.mkdir(parents=True,exist_ok=False)
    events=[];serial=0;started=time.monotonic();layout='service_island'
    conditions=['unchanged','rearrangement']
    def save(state='collecting_outcomes_unopened'):
        (root/'collection_status.json').write_text(json.dumps(dict(state=state,baseline_revision=revision,
            wall_s=time.monotonic()-started,events=events,planned_physical_runs=15),indent=2)+'\n')
    def job(condition,role,repeat,seed,extra):
        nonlocal serial
        serial+=1
        return dict(condition=condition,role=role,repeat=repeat,seed=seed,domain=170+serial,
            name=f'{layout}__{condition}__{role}__repeat{repeat}',extra=list(map(str,extra)))
    def run(j):
        verify(freeze)
        if '--session' in j['extra']:
            for name,digest in graph_hashes.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Graph lineage changed before loading')
        for attempt in range(2):
            out=root/(j['name']+('__setup_retry1' if attempt else ''))
            command=['/usr/bin/python3',str(freeze/'study/reserved_trial.py'),
                '--fixture-layout',layout,'--fixture-condition',j['condition'],'--split','reserved_correction_validation',
                '--frame-policy','surface_registration',
                '--out',str(out),'--seed',str(j['seed']),'--domain',str(j['domain']),'--wall-limit','620',*j['extra']]
            before=time.monotonic()
            log=root/(out.name+'.collection.log')
            with log.open('w') as stream:
                proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=proc.wait(timeout=720)
                except subprocess.TimeoutExpired:
                    proc.send_signal(signal.SIGINT)
                    try:proc.wait(timeout=45)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    code='outer_timeout'
            health=json.loads((out/'run.json').read_text()) if (out/'run.json').exists() else {}
            prephysics=(health.get('physics_started') is False)
            # A domain lock refusal occurs before output creation and before any physics.
            lock_refusal=(not out.exists() and 'is already in use' in log.read_text())
            event=dict(kind='process_health',job=j,artifact_directory=str(out),attempt=attempt,returncode=code,
                wall_s=time.monotonic()-before,physics_started=health.get('physics_started'),
                stop_reason=health.get('stop_reason'),has_map=(out/'mapper/map.txt').exists(),
                invalid_setup=prephysics or lock_refusal)
            events.append(event);save();print(json.dumps({k:event[k] for k in ['kind','artifact_directory','returncode','invalid_setup']}),flush=True)
            if attempt==0 and event['invalid_setup']:continue
            return out
    def stage(jobs):
        paths={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(run,j):j for j in jobs}
            for future in concurrent.futures.as_completed(futures):paths[futures[future]['name']]=future.result()
        return paths
    def qualify(path,frame,anchor_survey):
        try:
            fit=surface_audit(path,dock)
            selected=json.loads(fit.read_text())['target_from_source']
            frame.update(target_from_source=selected,map_from_mission=inverse(selected).tolist(),
                active_frame_source=str(fit),frame_policy='surface_registration')
            (path/'registration_frame.json').write_text(json.dumps(frame,indent=2)+'\n')
            result=analyze(fit,path/'source_surface.txt',path/'target_surface.txt',path,
                primary_transform=selected,anchor_survey_dir=anchor_survey,frame_policy='surface_fit')
            result['decision_time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            (path/'preacceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            return True
        except (ValueError,RuntimeError,OSError,StopIteration,subprocess.TimeoutExpired) as error:
            events.append(dict(kind='diagnostic_unavailable',run=path.name,reason='surface qualification unavailable',detail=str(error)));save()
            # Unavailable qualification cannot admit useful operation, but a
            # saved dock-anchored raster still receives a counterfactual audit
            # mission. Do not hide potentially unnecessary rejections.
            result=dict(decision_time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                diagnostic_error=str(error),primary_transform=frame['target_from_source'],
                rules={'operational':dict(decision='reject',reasons=['qualification_unavailable'],
                    calibrated_integrity=False,physical_mission_certified=False)})
            (path/'preacceptance.json').write_text(json.dumps(result,indent=2)+'\n')
            return (path/'mapper/map.txt').exists()
    commission=job('unchanged','commission',0,5900,['--mapping','new','--localizer','slam_toolbox',
        '--map-from-mission',*inverse(dock)])
    save();paths=stage([commission]);commission_path=paths[commission['name']]
    commission_frame=None;commission_ready=False
    try:
        commission_frame=anchor(commission_path,dock)
        commission_ready=qualify(commission_path,commission_frame,commission_path)
    except (ValueError,OSError,StopIteration) as error:
        events.append(dict(kind='dependency_failure',run=commission['name'],detail=str(error)));save()
    graph=commission_path/'mapper/session'
    graph_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [graph.with_suffix('.posegraph'),graph.with_suffix('.data')] if path.exists()}
    graph_ready=commission_ready and len(graph_hashes)==2
    (root/'commission_graph_hashes.json').write_text(json.dumps(graph_hashes,indent=2)+'\n')
    acquisition_jobs=[];bycondition={}
    for ci,condition in enumerate(conditions):
        bycondition[condition]=[]
        for repeat in range(2):
            j=job(condition,'full_acquisition',repeat,6000+ci*100+repeat,
                ['--mapping','new','--localizer','slam_toolbox','--map-from-mission',*inverse(dock)])
            acquisition_jobs.append(j);bycondition[condition].append(j)
        if graph_ready:
            acquisition_jobs.append(job(condition,'continued_acquisition',0,6000+ci*100,
                ['--mapping','continued','--localizer','slam_toolbox','--session',graph,
                 '--map-from-mission',*commission_frame['map_from_mission']]))
            g=job(condition,'graph_deployment',0,6000+ci*100,
                ['--mapping','localization','--localizer','slam_toolbox','--session',graph,
                 '--map-from-mission',*commission_frame['map_from_mission']])
            (root/(g['name']+'.preacceptance.json')).write_bytes((commission_path/'preacceptance.json').read_bytes())
            acquisition_jobs.append(g)
        else:events.append(dict(kind='dependency_failure',condition=condition,reason='historical graph unavailable'))
    paths=stage(acquisition_jobs);deployments=[]
    for j in acquisition_jobs:
        if not j['role'].endswith('acquisition'):continue
        path=paths[j['name']]
        if not (path/'mapper/map.txt').exists():events.append(dict(kind='dependency_failure',run=j['name'],reason='no candidate raster'));continue
        try:frame=anchor(path,dock) if j['role']=='full_acquisition' else dict(commission_frame)
        except (ValueError,OSError,StopIteration) as error:
            events.append(dict(kind='dependency_failure',run=j['name'],reason='initial gauge unavailable',detail=str(error)));continue
        inherited=j['role']=='continued_acquisition'
        if inherited:
            (path/'dock_anchor.json').write_text(json.dumps(dict(**frame,inherited_from=str(commission_path)),indent=2)+'\n')
            (path/'frame_lineage.json').write_text(json.dumps(dict(loaded_graph_hashes=graph_hashes,anchor_survey=str(commission_path)),indent=2)+'\n')
        if not qualify(path,frame,commission_path if inherited else path):continue
        deployments.append(job(j['condition'],j['role'].replace('acquisition','deployment'),j['repeat'],j['seed'],
            ['--remedy','acquired_map','--map',path/'mapper/map.txt','--map-from-mission',*frame['map_from_mission']]))
    save();stage(deployments)
    for name,digest in graph_hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Shared commissioning graph was modified')
    verify(freeze);save('collection_complete_outcomes_unopened')
    (root/'collection_gate.json').write_text(json.dumps(dict(allowed_to_open=True,
        note='All planned runs/dependencies complete; no physical outcome inspected during collection.'),indent=2)+'\n')
    print('Reserved collection complete; physical outcomes remain unopened.',flush=True)


if __name__=='__main__':main()
