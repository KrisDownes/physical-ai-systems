"""Bounded unchanged-site development gate; reserved layouts never generated."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time
from correction_pipeline import anchor,surface_audit
from frames import inverse
from transfer_layouts import _spec

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.out.resolve()
    root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for li,layout in enumerate(['maintenance_bays','l_workshop']):
        for repeat in range(2):
            jobs.append(dict(layout=layout,repeat=repeat,seed=5100+li*100+repeat,
                name=f'{layout}__unchanged__repeat{repeat}',domain=150+li*2+repeat))
    (root/'plan.json').write_text(json.dumps(dict(split='development',question='Do corrected unchanged controls repeat across independent mappings?',
        physical_runs=8,jobs=jobs,requirements='Unchanged15cm/10deg/2s/500simseconds,6orderedvisits+dock,no contact',
        gate='All four fresh missions pass; no deployment startup failure or unjudged map. Otherwise preserve reserved cohort.',
        reserved_cohort_generated=False),indent=2)+'\n')
    events=[]
    def run(job,role,extra):
        out=root/(job['name']+'__'+role)
        cmd=['/usr/bin/python3',str(ROOT/'research/corrected_trial.py'),'--fixture-layout',job['layout'],
            '--fixture-condition','unchanged','--out',str(out),'--seed',str(job['seed']),
            '--domain',str(job['domain']),'--wall-limit','620',*map(str,extra)]
        before=time.monotonic()
        with (root/(out.name+'.log')).open('w') as f:reply=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=720)
        event=dict(run=out.name,command=cmd,returncode=reply.returncode,wall_s=time.monotonic()-before)
        events.append(event);(root/'events.json').write_text(json.dumps(events,indent=2)+'\n');print(json.dumps(event),flush=True)
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run,j,'acquisition',['--mapping','new','--localizer','slam_toolbox',
            '--map-from-mission',*inverse(_spec(j['layout'])['start'])]):j for j in jobs}
        paths={}
        for future in concurrent.futures.as_completed(futures):paths[futures[future]['name']]=future.result()
    deployments=[]
    for j in jobs:
        path=paths[j['name']];dock=_spec(j['layout'])['start']
        if not (path/'controller_done.json').exists():raise RuntimeError('Unchanged survey did not complete; reserved cohort stays untouched')
        anc=anchor(path,dock);surface_audit(path,dock)
        deployments.append((j,path,anc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,j,'deployment',['--remedy','acquired_map','--map',str(path/'mapper/map.txt'),
            '--map-from-mission',*anc['map_from_mission']]) for j,path,anc in deployments]
        for future in concurrent.futures.as_completed(futures):future.result()
    print('Development collection complete; no reserved fixture was generated.',flush=True)


if __name__=='__main__':main()
