"""Bounded frozen cohort collection. Never reads physical outcome files."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import signal
import sys
import time

from transfer_trial import module,verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();freeze=a.freeze.resolve();root=a.out.resolve()
    root.mkdir(parents=True,exist_ok=False);revision=verify(freeze)
    generator=module('cohort_geometry',freeze/'study/transfer_layouts.py')
    sys.path.insert(0,str(freeze/'research'))
    from frames import inverse
    layouts=['maintenance_bays','l_workshop'];conditions=['unchanged','additions','rearrangement']
    events=[];started=time.monotonic();serial=0

    def save():
        (root/'collection_status.json').write_text(json.dumps(dict(
            baseline_revision=revision,state='collecting_without_outcome_inspection',
            wall_s=time.monotonic()-started,events=events),indent=2)+'\n')

    def trial(layout,condition,name,seed,extra):
        nonlocal serial
        serial+=1;domain=120+serial
        return dict(layout=layout,condition=condition,name=name,seed=seed,domain=domain,extra=list(map(str,extra)))

    def run(job):
        verify(freeze)
        path=root/job['name'];command=['/usr/bin/python3',str(freeze/'study/transfer_trial.py'),
            '--freeze',str(freeze),'--cohort-layout',job['layout'],'--cohort-condition',job['condition'],
            '--out',str(path),'--seed',str(job['seed']),'--domain',str(job['domain']),
            '--wall-limit','620',*job['extra']]
        before=time.monotonic()
        with (root/(job['name']+'.collection.log')).open('w') as f:
            try:
                proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                code=proc.wait(timeout=720)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=60)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                code='outer_timeout'
        # Only process/artifact health, not outcome.json or judge traces.
        return dict(job=job,returncode=code,wall_s=time.monotonic()-before,
            has_run_record=(path/'run.json').exists(),has_map=(path/'mapper/map.txt').exists(),
            has_graph=(path/'mapper/session.posegraph').exists(),kind='process_health')

    def stage(jobs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(run,j):j for j in jobs}
            for future in concurrent.futures.as_completed(futures):
                event=future.result();events.append(event);save()
                print(json.dumps(dict(completed=event['job']['name'],returncode=event['returncode'],
                    total_completed=len([e for e in events if e['kind']=='process_health']))),flush=True)

    def align(path,prior):
        output=path/'alignment.json';before=time.monotonic()
        command=['/usr/bin/python3',str(freeze/'research/register_maps.py'),
            '--source',str(path/'mapper/map.txt'),'--target',str(path/'map.txt'),
            '--output',str(output),'--prior',*map(str,prior),
            '--method','robust','--selection-objective','trimmed','--max-correspondence','.15',
            '--trim-fraction','.7','--huber-delta','.08','--inlier-distance','.12',
            '--translation-radius','.25','--yaw-radius','.1','--max-points','12000',
            '--score-cap','.3','--iterations','100','--translation-tolerance','.00001','--yaw-tolerance','.00001']
        with (path/'registration.log').open('w') as f:
            result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=60)
        events.append(dict(kind='registration',run=path.name,returncode=result.returncode,wall_s=time.monotonic()-before));save()
        if result.returncode:return None
        fit=json.loads(output.read_text());pose=fit['target_from_source']
        if len(pose)!=3:raise ValueError('Malformed registration')
        # Diagnostics are computed and timestamped BEFORE the fresh mission.
        with (path/'preacceptance.log').open('w') as f:
            subprocess.run(['/usr/bin/python3',str(freeze/'study/transfer_acceptance.py'),
                '--alignment',str(output),'--source',str(path/'mapper/map.txt'),
                '--target',str(path/'map.txt'),'--output',str(path/'preacceptance.json')],
                stdout=f,stderr=subprocess.STDOUT,timeout=60,check=True)
        return inverse(pose).tolist()

    commission=[]
    for li,layout in enumerate(layouts):
        start=generator._spec(layout)['start']
        commission.append(trial(layout,'unchanged',layout+'__commission',900+li,
            ['--mapping','new','--localizer','slam_toolbox','--map-from-mission',*inverse(start)]))
    save();stage(commission)
    transforms={}
    for layout in layouts:
        path=root/(layout+'__commission')
        if (path/'mapper/map.txt').exists() and (path/'mapper/session.posegraph').exists():
            transforms[layout]=align(path,generator._spec(layout)['start'])

    acquisitions=[]
    for ci,condition in enumerate(conditions):
        for li,layout in enumerate(layouts):
            prefix=layout+'__'+condition
            acquisitions.append(trial(layout,condition,prefix+'__full_acquisition',1000+li*10+ci,
                ['--mapping','new','--localizer','slam_toolbox','--map-from-mission',*inverse(generator._spec(layout)['start'])]))
            if transforms.get(layout) is not None:
                graph=root/(layout+'__commission')/'mapper/session'
                acquisitions.append(trial(layout,condition,prefix+'__continued_acquisition',2000+li*10+ci,
                    ['--mapping','continued','--localizer','slam_toolbox','--session',graph,
                     '--map-from-mission',*transforms[layout]]))
                acquisitions.append(trial(layout,condition,prefix+'__graph_deployment',3000+li*100+ci*10,
                    ['--mapping','localization','--localizer','slam_toolbox','--session',graph,
                     '--map-from-mission',*transforms[layout]]))
            else:events.append(dict(kind='dependency_missing',layout=layout,condition=condition,reason='commissioning map/graph/alignment unavailable; no replacement'))
    stage(acquisitions)
    deployments=[]
    for ci,condition in enumerate(conditions):
        for li,layout in enumerate(layouts):
            prefix=layout+'__'+condition
            for method in ['full','continued']:
                source=root/(prefix+'__'+method+'_acquisition')
                if not (source/'mapper/map.txt').exists():
                    events.append(dict(kind='dependency_missing',run=source.name,reason='no candidate map; workflow failure retained'));continue
                pose=align(source,generator._spec(layout)['start']) if method=='full' else transforms.get(layout)
                if pose is None:continue
                deployments.append(trial(layout,condition,prefix+'__'+method+'_deployment',3000+li*100+ci*10,
                    ['--remedy','acquired_map','--map',source/'mapper/map.txt','--map-from-mission',*pose]))
    stage(deployments);verify(freeze)
    record=json.loads((root/'collection_status.json').read_text());record.update(
        state='collection_complete_outcomes_not_inspected',wall_s=time.monotonic()-started,
        planned_launches=32,actual_launches=serial,events=events)
    (root/'collection_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Collection complete. No physical outcome was inspected by this driver.',flush=True)


if __name__=='__main__':main()
