"""Unique experiment directory with project and upstream provenance."""
import gzip,hashlib,json,os,platform,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/published_runs'
UPSTREAM='cf2cb670fb983f67cf2af156d4e17b4a690787e2'
ARCHIVE='caa46b6e0925370d4b005eb7d574a7169cb7ef27eb615eb4f43c50470a35b826'
def create(label,config):
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');path=OUT/f'{now}_{revision[:12]}_{label}_{uuid.uuid4().hex[:8]}';path.mkdir(parents=True,exist_ok=False)
 files=sorted((ROOT/'published_manipulation').glob('*.py'))+sorted((ROOT/'published_manipulation').glob('*.json'))
 source={str(p.relative_to(ROOT)):p.read_text() for p in files}
 with gzip.open(path/'source_snapshot.json.gz','wt') as f:json.dump(source,f,sort_keys=True)
 freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
 (path/'dependencies.txt').write_text(freeze)
 meta={'revision':revision,'created_utc':now,'label':label,'config':config,'python':sys.version,'platform':platform.platform(),'upstream_revision':UPSTREAM,'upstream_archive_sha256':ARCHIVE,'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()},'git_status_project':subprocess.check_output(['git','status','--short','--','published_manipulation'],cwd=ROOT,text=True)}
 (path/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return path
