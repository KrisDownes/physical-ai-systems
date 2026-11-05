"""Complete only proven pre-ROS argparse failures and their planned dependencies."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();freeze=a.freeze.resolve();root=a.root.resolve();dest=root/'transport_recovery';dest.mkdir(exist_ok=False)
    sys.path.insert(0,str(freeze/'study'));from transfer_trial import verify
    sys.path.insert(0,str(freeze/'research'));from frames import inverse
    verify(freeze)
    fix=json.loads((freeze/'recovery_v1_1/manifest.json').read_text())
    for name,digest in fix['files'].items():assert hashlib.sha256((freeze/'recovery_v1_1'/name).read_bytes()).hexdigest()==digest
    status=json.loads((root/'collection_status.json').read_text())
    assert status['state']=='collection_complete_outcomes_not_inspected' and not (root/'outcomes_opened.json').exists()
    original_events=list(status['events']);records=[];serial=0
    invalid=[]
    for e in original_events:
        if e['kind']!='process_health' or e['returncode']!=2:continue
        name=e['job']['name'];log=root/(name+'.collection.log')
        if 'argument --map-from-mission: expected 3 arguments' not in log.read_text():continue
        assert not (root/name).exists(),'Not a proven pre-fixture failure'
        invalid.append(e)
    (dest/'invalid_setup_attempts.json').write_text(json.dumps(dict(
        classification='pre-ROS parser transport failure, not a physical method failure',
        original_attempts=invalid,simulator_resets_for_invalid_attempts=0),indent=2)+'\n')
    def job(base):
        nonlocal serial
        serial+=1;new=dict(base);new['domain']=200+serial;return new
    def execute(j):
        verify(freeze);name=j['name'];before=time.monotonic()
        cmd=['/usr/bin/python3',str(freeze/'recovery_v1_1/transfer_trial_fixed.py'),
            '--freeze',str(freeze),'--cohort-layout',j['layout'],'--cohort-condition',j['condition'],
            '--out',str(dest/name),'--seed',str(j['seed']),'--domain',str(j['domain']),
            '--wall-limit','620',*j['extra']]
        with (dest/(name+'.collection.log')).open('w') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=720)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=60)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code='outer_timeout'
        return dict(kind='process_health',job=j,returncode=code,wall_s=time.monotonic()-before,
            artifact_directory=str(dest/name),setup_recovery=True,
            has_run_record=(dest/name/'run.json').exists(),has_map=(dest/name/'mapper/map.txt').exists(),
            has_graph=(dest/name/'mapper/session.posegraph').exists())
    def stage(jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for value in pool.map(execute,jobs):
                records.append(value);(dest/'process_health.json').write_text(json.dumps(records,indent=2)+'\n')
                print(json.dumps(dict(recovered=value['job']['name'],returncode=value['returncode'])),flush=True)
    stage([job(e['job']) for e in invalid])
    # Complete only originally planned fresh deployments whose prerequisite
    # acquisition never started due to the allowlisted numeric transport error.
    pending=[]
    for e in invalid:
        old=e['job'];name=old['name']
        if not name.endswith('__continued_acquisition'):continue
        source=dest/name
        if not (source/'mapper/map.txt').exists():continue
        layout=old['layout'];condition=old['condition'];deployment=name.replace('__continued_acquisition','__continued_deployment')
        assert not (root/deployment).exists() and not (dest/deployment).exists()
        fit=json.loads((root/(layout+'__commission')/'alignment.json').read_text());pose=inverse(fit['target_from_source']).tolist()
        li=['maintenance_bays','l_workshop'].index(layout);ci=['unchanged','additions','rearrangement'].index(condition)
        pending.append(job(dict(layout=layout,condition=condition,name=deployment,seed=3000+li*100+ci*10,
            extra=['--remedy','acquired_map','--map',str(source/'mapper/map.txt'),'--map-from-mission',*map(str,pose)])))
    stage(pending);verify(freeze)
    invalid_names={e['job']['name'] for e in invalid}
    # Keep original failures in events. Mark the specific missing dependency as
    # resolved only if the planned candidate/deployment now has an execution.
    for e in status['events']:
        if e['kind']=='dependency_missing' and e.get('run') in invalid_names:
            e['transport_recovery_record']=str(dest/e['run'])
    status['events'].extend(records);status['actual_launches']+=len(records)
    status['transport_invalid_launches']=len(invalid);status['transport_recovery_manifest']=fix
    status['valid_execution_attempts']=sum(e.get('has_run_record',False) for e in status['events'] if e['kind']=='process_health')
    (root/'collection_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (root/'collection_gate.json').write_text(json.dumps(dict(allowed_to_open=True,
        note='All allowlisted setup retries and planned dependencies completed without physical-outcome inspection.'),indent=2)+'\n')
    print('Transport recovery complete; physical outcomes remain unopened.',flush=True)


if __name__=='__main__':main()
