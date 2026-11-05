"""Freeze the camera-belief arm on three already-open diagnostic checkpoints."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
OUT = ROOT/'results/inspection_aware_20261028'
BASE = ROOT/'results/matched_visibility_20261027'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((BASE/'manifest_v3.json').read_text())
    for case in ('wrong_asset', 'moved', 'absent'):
        public = OUT/'templates'/case/'inspection/public'
        policy = json.loads((public/'policy_nearest.json').read_text())
        policy['search_selector'] = 'camera_belief'
        (public/'policy_aware.json').write_text(json.dumps(policy, indent=2)+'\n')
        assert sha(OUT/'templates'/case/'inspection/map.txt') == original['template_audit'][case]['checkpoint_map_sha256']
    sources = [
        ROOT/'research/inspection_aware'/name for name in
        ('belief_search.py', 'inspection.py', 'run_matched.py', 'prepare_matched.py')]
    sources += [ROOT/'research/continuous_asset'/name for name in
                ('inspection.py', 'visibility_search.py', 'buffered_planning.py', 'inspection_judge.py')]
    sources += [ROOT/'research/moved_asset'/name for name in
                ('planning.py', 'camera.py', 'render.py')]
    sources += [ROOT/'research/autonomous_asset/vision.py', ROOT/'research/controller.py',
                ROOT/'sim/judge_node.py', ROOT/'vendor/offline_amcl',
                ROOT/'docs/inspection_aware/protocol.md']
    sources += [p for p in (OUT/'templates').rglob('*') if p.is_file()]
    manifest = dict(source_cohort=str(BASE), cases=['wrong_asset', 'moved', 'absent'],
                    execution_order=[['wrong_asset','aware'],['moved','aware'],['absent','aware']],
                    diagnostic_reset_per_arm=1, continuous_mission_claim=False,
                    wall_guard_s=800,
                    template_audit=original['template_audit'],
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    files={str(p.relative_to(REPO)):sha(p) for p in sources},
                    outcomes_unopened_at_freeze=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(OUT/'manifest.json')


if __name__ == '__main__':
    main()
