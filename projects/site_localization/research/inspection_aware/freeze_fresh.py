"""Record all prior frozen dependencies plus scoped camera-search additions."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
OUT = ROOT/'results/fresh_inspection_aware_20261028'
PRIOR = ROOT/'results/fresh_visibility_20261027/pipeline_freeze.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    earlier = json.loads(PRIOR.read_text())
    names = set(earlier['files'])
    names.discard('projects/site_localization/docs/post_repair_connected/fresh_protocol.md')
    names.update(str((ROOT/x).relative_to(REPO)) for x in (
        'docs/inspection_aware/protocol.md',
        'research/inspection_aware/belief_search.py',
        'research/inspection_aware/inspection.py',
        'research/inspection_aware/prepare_fresh.py',
        'research/inspection_aware/execute_fresh.py',
        'research/inspection_aware/run_continuous.py',
        'research/inspection_aware/freeze_fresh.py'))
    record = dict(role='prospective pipeline freeze', method='three search selectors',
                  seed=7601, known_initial_dock=True, outcomes_unopened=True,
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  files={name:sha(REPO/name) for name in sorted(names)})
    (OUT/'pipeline_freeze.json').write_text(json.dumps(record,indent=2)+'\n')
    print(len(record['files']), OUT/'pipeline_freeze.json')


if __name__ == '__main__':
    main()
