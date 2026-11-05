"""Copy opened case templates and freeze the one-field search-policy change."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path(__file__).resolve().parents[2]
repo=root.parents[1]
old=root/'results/connected_layout_20261025/templates'
out=root/'results/visibility_search_20261027'
if out.exists():raise FileExistsError(out)
out.mkdir();(out/'templates').mkdir()
cases=['wrong_asset','moved','absent']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for case in cases:
    src=old/case;dst=out/'templates'/case
    shutil.copytree(src,dst)
    policy=dst/'public/policy.json'
    data=json.loads(policy.read_text())
    assert 'search_selector' not in data
    data['search_selector']='visibility_gain'
    policy.write_text(json.dumps(data,indent=2)+'\n')
    assert {k:v for k,v in json.loads(policy.read_text()).items() if k!='search_selector'}==json.loads((src/'public/policy.json').read_text())
    for p in src.rglob('*'):
        if not p.is_file():continue
        if p.relative_to(src)!=Path('public/policy.json'):
            assert sha(p)==sha(dst/p.relative_to(src)),p
files=[*sorted((root/'research/continuous_asset').glob('*.py'))]
files += [root/'research/autonomous_asset/pose.py',root/'research/autonomous_asset/vision.py']
files += [*sorted((root/'research/moved_asset').glob('*.py'))]
files += [root/'research/slam_bridge.py',root/'research/frames.py',root/'research/readiness.py',
          root/'research/correction_pipeline.py',root/'research/constraint_repair/slam_params_radius_1p5.yaml',
          root/'vendor/offline_amcl',root/'docs/post_repair_connected/visibility_trial_protocol.md']
files += [p for p in out.rglob('*') if p.is_file()]
files=list(dict.fromkeys(files))
manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              cases=cases,seed=7401,wall_limit_s=800,one_attempt_each=True,
              source_cohort='opened connected_layout_20261025',
              sole_template_difference='public/policy.json adds search_selector=visibility_gain',
              frozen_config='research/constraint_repair/slam_params_radius_1p5.yaml',
              method_source_change='optional conventional visibility_search selection; default nearest branch preserved',
              files={str(p.relative_to(repo)):sha(p) for p in files},outcomes_unopened=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out/'manifest.json',sha(out/'manifest.json'),len(files))
