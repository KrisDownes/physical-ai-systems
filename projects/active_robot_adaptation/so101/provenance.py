"""Record exact local source and external model hashes for this run."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

files=sorted([*Path('so101').glob('*.py'),*Path('so101').glob('*.json'),
              Path('so101/scene.xml'),Path('so101/fetch_model.sh'),Path('requirements-so101.lock.txt')])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'local_code_uncommitted_at_run':True,'model_revision':'370c43965323c0944b433b202aac5058b5f907a9',
        'model_sha256':sha(Path('.so101-model/so101.xml')),
        'source_sha256':{str(p):sha(p) for p in files},
        'python':platform.python_version(),'numpy':importlib.metadata.version('numpy'),
        'mujoco':importlib.metadata.version('mujoco')}
Path('results/so101/provenance.json').write_text(json.dumps(result,indent=2)+'\n')
