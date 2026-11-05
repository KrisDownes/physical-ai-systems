"""Run the three predeclared opened cases sequentially; keep every attempt."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
repo=root.parents[1]
out=root/'results/post_repair_connected_20261027'
manifest=json.loads((out/'manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in manifest['files'].items():
    path=repo/name
    if sha(path)!=expected:
        raise RuntimeError(f'Frozen input changed: {path}')
for case in ['occluded','wrong_asset','absent']:
    dest=out/f'mission_{case}'
    if dest.exists():
        raise FileExistsError(dest)
    command=['/usr/bin/python3',str(root/'research/continuous_asset/run.py'),
             '--template',str(root/'results/connected_layout_20261025/templates'/case),
             '--out',str(dest),'--slam-params',str(root/'research/constraint_repair/slam_params_radius_1p5.yaml'),
             '--seed','7401','--wall-limit','800']
    with (out/'execution.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='start',case=case,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command))+'\n')
    with (out/f'{case}.stdout').open('x') as f:
        code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
    outcome=json.loads((dest/'outcome.json').read_text()) if (dest/'outcome.json').exists() else {}
    with (out/'execution.jsonl').open('a') as f:
        f.write(json.dumps(dict(event='finish',case=case,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                returncode=code,complete_contract=outcome.get('complete_contract'),
                                infrastructure_failure=outcome.get('infrastructure_failure')))+'\n')
    print(case,code,outcome.get('complete_contract'),outcome.get('inspection_reason'),flush=True)
    if code or outcome.get('infrastructure_failure'):
        sys.exit(2)
