"""Create immutable-by-default, uniquely named experiment output directories."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('results/so101_runs')


def create_run(label, config):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    path=ROOT/f'{timestamp}_{revision[:12]}_{label}_{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True,exist_ok=False)
    files=sorted(Path('so101').glob('*.py'))+sorted(Path('so101').glob('*.json'))+[Path('so101/scene.xml')]
    metadata={'label':label,'revision':revision,'created_utc':timestamp,'config':config,
              'python':platform.python_version(),
              'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','mujoco')},
              'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              'git_project_status':subprocess.check_output(['git','status','--short','--','so101'],text=True).strip()}
    (path/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return path
