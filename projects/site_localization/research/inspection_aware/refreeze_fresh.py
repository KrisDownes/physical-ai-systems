"""Pre-outcome launcher-only revision; preserves initial freeze and scenarios."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
OUT = ROOT/'results/fresh_inspection_aware_20261028'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    earlier = json.loads((OUT/'pipeline_freeze.json').read_text())
    scenario = json.loads((OUT/'scenario_freeze.json').read_text())
    assert not list(OUT.glob('mission_*')), 'Fresh outcomes already opened'
    names = set(earlier['files'])
    names.update(str((ROOT/x).relative_to(REPO)) for x in (
        'research/inspection_aware/run_continuous.py',
        'research/inspection_aware/refreeze_fresh.py'))
    record = dict(earlier, revision='v2: separate experimental launcher; conventional run.py exact hash restored',
                  files={name:sha(REPO/name) for name in sorted(names)}, outcomes_unopened=True)
    save(OUT/'pipeline_freeze_v2.json', record)
    scenario.update(revision='v2: launcher-only freeze, same templates/order/seed',
                    pipeline_freeze_sha256=sha(OUT/'pipeline_freeze_v2.json'),
                    prior_pipeline_freeze_sha256=sha(OUT/'pipeline_freeze.json'),
                    outcomes_unopened_at_freeze=True)
    save(OUT/'scenario_freeze_v2.json', scenario)
    print(len(record['files']), OUT/'scenario_freeze_v2.json')


if __name__ == '__main__':
    main()
