"""Freeze after development; run each predeclared fresh case once, sequentially."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(root,execute):
    manifest=root/'final_freeze.json'
    if not execute:
        if manifest.exists():raise FileExistsError(manifest)
        for name in ['dev_return_unchanged_v2','dev_return_moved_v2','dev_continuous_moved_v2']:
            result=load(root/name/'outcome.json')
            if not result['complete_contract']:raise RuntimeError('Development prerequisite not met: '+name)
        files={}
        for name in ['dev_return_moved_v2','dev_continuous_moved_v2']:
            for p in load(root/name/'freeze.json')['files']:files[p]=sha(Path(p))
        for p in (root/'scenarios/fresh').rglob('*'):
            if p.is_file():files[str(p.resolve())]=sha(p)
        source=load(root/'scenarios/scenario_freeze.json')
        for p,h in source['files'].items():assert sha(Path(p))==h,p
        frozen=dict(revision='continuous-v2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,
            cases=['unchanged','moved','wrong_asset','absent'],seed=7401,wall_limit_s=800,
            source_scenarios_sha256=sha(root/'scenarios/scenario_freeze.json'),development_prerequisites=['dev_return_unchanged_v2','dev_return_moved_v2','dev_continuous_moved_v2'],
            repeat_policy='No automatic reruns; every attempted directory and result retained; infrastructure failure stops',
            primary='complete physical mission from initial dock through final termination, no intervening reset')
        manifest.write_text(json.dumps(frozen,indent=2)+'\n');print('Frozen',sha(manifest));return
    frozen=load(manifest)
    for case in frozen['cases']:
        for p,h in frozen['files'].items():assert sha(Path(p))==h,p
        out=root/('fresh_'+case)
        if out.exists():raise FileExistsError('Never replace an attempted execution: '+str(out))
        command=['/usr/bin/python3',str(HERE/'run.py'),'--template',str(root/'scenarios/fresh'/case),'--out',str(out),'--seed',str(frozen['seed']),'--wall-limit',str(frozen['wall_limit_s'])]
        with (root/('fresh_'+case+'.log')).open('x') as f:code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Run/scoring failure retained: '+case)
        result=load(out/'outcome.json');print(case,result['complete_contract'],result['termination'],flush=True)
        if result['infrastructure_failure']:raise RuntimeError('Infrastructure failure retained; stop before any rerun')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.root.resolve(),a.execute)
