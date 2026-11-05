"""Freeze an opened development cohort before its corrected closed-loop runs."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
repo=root.parents[1]
out=root/'results/post_repair_connected_20261027'
if out.exists():
    raise FileExistsError(out)
out.mkdir()
cases=['visible','moved','occluded','wrong_asset','absent']
source=[*sorted((root/'research/continuous_asset').glob('*.py'))]
source += [root/'research/autonomous_asset/pose.py',root/'research/autonomous_asset/vision.py']
source += [*sorted((root/'research/moved_asset').glob('*.py'))]
source += [root/'research/slam_bridge.py',root/'research/frames.py',root/'research/readiness.py',root/'research/correction_pipeline.py']
source += [root/'research/constraint_repair/slam_params_radius_1p5.yaml',root/'vendor/offline_amcl',
           root/'docs/post_repair_connected/protocol.md']
templates=root/'results/connected_layout_20261025/templates'
source += [p for case in cases for p in sorted((templates/case).rglob('*')) if p.is_file()]
source=list(dict.fromkeys(source))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              protocol='docs/post_repair_connected/protocol.md',
              prior_connected_evaluation_sha256=sha(root/'results/connected_layout_20261025/execution_freeze.json'),
              radius_config_sha256=sha(root/'research/constraint_repair/slam_params_radius_1p5.yaml'),
              cases=cases,seed=7401,wall_limit_s=800,
              stop_after_first_two_if_mapping_or_handoff_grossly_wrong=True,
              attempts='one per case; infrastructure-only rerun separately documented',
              outcomes_unopened_in_new_cohort=True,
              files={str(p.relative_to(repo)):sha(p) for p in source})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out/'manifest.json',sha(out/'manifest.json'),len(source))
