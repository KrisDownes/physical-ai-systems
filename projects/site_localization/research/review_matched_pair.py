"""Fixed saved-candidate comparison; prepare first, then execute sequentially.

No optimizer is called and no pipeline default is modified. Selection is an
explicit recorded objective over the original 27 immutable candidate records.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from frames import compose, inverse
from transfer_layouts import make

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REVIEW = ROOT/'results/technical_review_20261023'
HISTORICAL = ROOT/'results/correction_surface_policy_20261022/maintenance_bays__unchanged__repeat1__deployment'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def select_saved(alignment, objective, expected_index):
    """Conventional scoring choice only; keep optimizer and all candidates."""
    key = {'trimmed': 'trimmed_objective_m2', 'capped': 'objective_m2'}[objective]
    index = min(range(len(alignment['candidates'])), key=lambda i: alignment['candidates'][i][key])
    if index != expected_index:
        raise ValueError(f'Fixed review candidate changed: {objective}, {index}')
    return dict(selection_objective=objective, selected_score_field=key,
        selected_index=index, selected=alignment['candidates'][index],
        parameters=alignment['parameters'], original_candidates=alignment['candidates'],
        scoring_scope='saved source-to-target distances; all-source cap 0.30m or fixed 70% trimmed loss',
        candidate_optimization_rerun=False, additional_inputs=False,
        note='Diagnostic calibration choice; not a deployment acceptance rule.')


def prepare(out):
    cpath = REVIEW/'next_candidate.json'
    c = load(cpath)
    manifest = load(REVIEW/'review_manifest.json')
    assert sha(cpath) == manifest['files']['results/technical_review_20261023/next_candidate.json']
    alignment_path, map_path = REPO/c['source_alignment'], REPO/c['map']
    assert sha(alignment_path) == c['source_alignment_sha256']
    assert sha(map_path) == c['map_sha256']
    a = load(alignment_path)
    assert len(a['candidates']) == 27 and c['seed'] == 5101
    assert a['parameters']['score_cap'] == .3 and a['parameters']['trim_fraction'] == .7
    assert a['candidates'][0]['pose'] == c['baseline_target_from_source']
    assert a['candidates'][3]['pose'] == c['target_from_source']
    assert inverse(c['target_from_source']).tolist() == c['map_from_mission']
    assert inverse(c['baseline_target_from_source']).tolist() == c['baseline_map_from_mission']
    files = [ROOT/'research'/name for name in ['run.py','controller.py','frames.py','scenarios.py','score.py']]
    files += [ROOT/'sim/judge_node.py', ROOT/'sim/score_mission.py', ROOT/'vendor/offline_amcl',
              ROOT/'vendor/offline_amcl.cpp', ROOT/'research/corrected_trial.py', ROOT/'research/transfer_layouts.py',
              ROOT/'sim/cell.sdf', ROOT/'vendor/setup_amcl.bash', Path(__file__)]
    # Core runtime behavior must match the archived original failure.
    for p in files[:7]:
        assert sha(p) == sha(HISTORICAL/'source'/p.name), p
    old_hashes = load(HISTORICAL/'source_hashes.json')
    assert sha(ROOT/'vendor/offline_amcl') == old_hashes[str(ROOT/'vendor/offline_amcl')]
    out.mkdir(parents=True, exist_ok=False)
    (out/'source').mkdir()
    for p in files:
        if p.suffix in ['.py', '.sdf', '.bash', '.cpp']:
            (out/'source'/p.name).write_bytes(p.read_bytes())
    (out/'original_alignment.json').write_bytes(alignment_path.read_bytes())
    write(out/'candidate_verification.json', dict(review_candidate=c, verified_utc=now(),
        review_candidate_sha256=sha(cpath), alignment_sha256=sha(alignment_path), map_sha256=sha(map_path),
        historical_runtime_matches=True, source_hashes={str(p):sha(p) for p in files}))
    jobs, configs = [], []
    historical_cfg = load(HISTORICAL/'controller_config.json')
    for objective, index in [('trimmed', 0), ('capped', 3)]:
        selection = select_saved(a, objective, index)
        pose = selection['selected']['pose']
        transform = inverse(pose).tolist()
        prepared = out/('prepared_'+objective)
        cfg = make(prepared, 'maintenance_bays', 'unchanged', 'acquired_map')
        cfg.update(control_dwell_margin_s=.2, arrival_tolerance_m=.03, slam_pose_source='tf',
                   localizer='amcl', seed=5101, mapping='off', map_from_mission=transform,
                   map_source=str(map_path))
        keys = historical_cfg.keys()
        controller_cfg = {k:cfg[k] for k in keys}
        assert {k:v for k,v in controller_cfg.items() if k!='map_from_mission'} == {
            k:v for k,v in historical_cfg.items() if k!='map_from_mission'}
        write(prepared/'expected_cell.json', cfg)
        write(prepared/'expected_controller_config.json', controller_cfg)
        selection.update(target_from_source=pose, map_from_mission=transform,
            map_initial_pose=compose(transform, cfg['start']).tolist(),
            map_route_poses=compose(transform, np.array([r['pose'] for r in cfg['route']])).tolist(),
            map_inspection_poses=compose(transform, np.array(cfg['goals'])).tolist(),
            immutable_site_goals=cfg['goals'], source_alignment_sha256=sha(alignment_path),
            map_sha256=sha(map_path))
        write(prepared/'calibration_selection.json', selection)
        mission_dir = out/('maintenance_bays__unchanged__'+objective+'__deployment')
        command = ['/usr/bin/python3', str(ROOT/'research/corrected_trial.py'),
            '--fixture-layout','maintenance_bays','--fixture-condition','unchanged',
            '--split','review_matched_development','--frame-policy','surface_fit_diagnostic',
            '--out',str(mission_dir),'--seed','5101','--domain','199','--wall-limit','620',
            '--remedy','acquired_map','--map',str(map_path),'--map-from-mission',*map(str,transform)]
        jobs.append(dict(objective=objective,candidate_index=index,output=str(mission_dir),
                         prepared=str(prepared),command=command))
        configs.append(controller_cfg)
    assert {k:v for k,v in configs[0].items() if k!='map_from_mission'} == {
        k:v for k,v in configs[1].items() if k!='map_from_mission'}
    assert sha(out/'prepared_trimmed/cell.sdf') == sha(out/'prepared_capped/cell.sdf') == sha(HISTORICAL/'cell.sdf')
    write(out/'plan.json', dict(registered_utc=now(), jobs=jobs, seed=5101,
        same_ros_domain_reused_sequentially=199, simultaneous_missions=1,
        intended_treatment='Only candidate scoring choice and selected T_site_map; inverse transforms BOTH initial mean and route/inspection/dock goals in map coordinates.',
        input_equivalence='Controller config differs only in map_from_mission; generated world identical to each other and original failure; same source, map, seed, sensor and particle settings.',
        procedural_differences='Fresh resets and unique output paths/Gazebo partitions. Process scheduling/measurement arrival are not deterministic or forced identical.',
        fixed_physical_contract=c['physical_criteria'], wall_guard_s=620,
        assistance='Same ideal commissioning geometry, known dock, zero planar lidar extrinsics and preplanned route. No new anchors.',
        stopping_rule='Two sequential attempts. Physical failures retained. Stop for diagnosis on unscored/no-physics infrastructure failure; no automatic retry or outcome-selected replacement.',
        outcome_interpretation='Original fails/capped succeeds: resolves this development failure. Original passes: diagnose reproducibility before effect claim. Capped fails: earliest divergence/error decomposition. No reliability/generalization claim.',
        reserved_access=False))
    print('Verified and prepared exactly two fixed missions:', out, flush=True)


def execute(out):
    plan = load(out/'plan.json')
    if (out/'attempts.json').exists():
        raise FileExistsError('Attempts already recorded; do not silently rerun')
    for p,h in load(out/'candidate_verification.json')['source_hashes'].items():
        assert sha(p) == h, p
    attempts = []
    for job in plan['jobs']:
        path, prep = Path(job['output']), Path(job['prepared'])
        assert not path.exists()
        event = dict(attempt=len(attempts)+1, objective=job['objective'], command=job['command'],
                     output=str(path), started_utc=now(), state='launching', reset_requested=1)
        attempts.append(event); write(out/'attempts.json', attempts)
        start = time.monotonic()
        with (out/(job['objective']+'.log')).open('w') as log:
            proc = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            event.update(pid=proc.pid,state='running');write(out/'attempts.json',attempts)
            try:
                event['returncode'] = proc.wait(timeout=750)
            except subprocess.TimeoutExpired:
                event['infrastructure_issue'] = '750 second parent timeout; sent SIGINT for launcher cleanup'
                os.killpg(proc.pid,signal.SIGINT)
                event['returncode'] = proc.wait(timeout=60)
        event.update(elapsed_parent_wall_s=time.monotonic()-start, finished_utc=now(),state='finished')
        if path.exists():
            (path/'calibration_selection.json').write_bytes((prep/'calibration_selection.json').read_bytes())
            actual = load(path/'controller_config.json')
            assert actual == load(prep/'expected_controller_config.json')
            assert sha(path/'map.txt') == load(path/'calibration_selection.json')['map_sha256']
            assert sha(path/'cell.sdf') == sha(prep/'cell.sdf')
        run = load(path/'run.json') if (path/'run.json').exists() else {}
        event.update(physics_started=run.get('physics_started'), reset_count=run.get('reset_count'),
            all_owned_processes_reaped=run.get('all_owned_processes_reaped'),stop_reason=run.get('stop_reason'),
            score_returncode=run.get('score_returncode'))
        write(out/'attempts.json',attempts)
        print(json.dumps(event),flush=True)
        if not (run.get('physics_started') and run.get('all_owned_processes_reaped') and run.get('score_returncode') == 0):
            raise RuntimeError('Infrastructure/scoring failure retained. Diagnose before any next attempt.')
    write(out/'collection_complete.json',dict(finished_utc=now(),attempts=len(attempts),sequential=True,
        no_retries=True,reset_count=sum(e['reset_count'] for e in attempts),reserved_access=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    (execute if args.execute else prepare)(args.out.resolve())
