"""Post-run evaluator analysis and demonstration; never imported by the robot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_pose(path):
    with path.open() as stream:
        data = list(csv.DictReader(stream))
    return np.array([[float(row[k]) for k in ['timestamp', 'x', 'y', 'yaw']] for row in data])


def truth_pose(path):
    data = [r for r in rows(path / 'judge.jsonl') if r['type'] == 'truth']
    return np.array([[r['time'], r['x'], r['y'],
                      np.arctan2(2 * (r['q'][3] * r['q'][2] + r['q'][0] * r['q'][1]),
                                 1 - 2 * (r['q'][1] ** 2 + r['q'][2] ** 2))] for r in data])


def error_summary(estimate, truth):
    reference = np.column_stack([np.interp(estimate[:, 0], truth[:, 0], truth[:, k])
                                 for k in [1, 2]])
    error = np.linalg.norm(estimate[:, 1:3] - reference, axis=1)
    heading = estimate[:, 3] - np.interp(estimate[:, 0], truth[:, 0], np.unwrap(truth[:, 3]))
    heading = np.degrees(abs(np.arctan2(np.sin(heading), np.cos(heading))))
    return dict(samples=len(error), position_median_m=float(np.median(error)),
                position_p95_m=float(np.percentile(error, 95)), position_max_m=float(error.max()),
                heading_p95_deg=float(np.percentile(heading, 95)),
                note='Descriptive within-trajectory errors, not independent trials')


def analyze(root):
    analysis = root / 'analysis'
    analysis.mkdir(exist_ok=True)
    ledger, diagnostics = [], {}
    starts = [r for r in rows(root / 'attempts.jsonl') if r['event'] == 'attempt_started']
    for start in starts:
        out = Path(start['output'])
        result, run = load(out / 'outcome.json'), load(out / 'run.json')
        acq = result.get('acquisition', {})
        entry = dict(attempt=out.name, cohort='fresh' if out.name.startswith('fresh_') else 'development',
                     case=result['case'], return_required=result['return_required'],
                     complete_contract=result['complete_contract'], honest_noncompletion=result.get('honest_noncompletion', False),
                     false_completion=result['false_completion'], wrong_asset_acceptances=result['wrong_asset_acceptances'],
                     handoff_success=result['handoff_success'], contacts=result['contacts'], clearance_m=result['clearance_m'],
                     total_sim_s=result['total_sim_s'], total_path_m=result['total_path_m'],
                     acquisition_sim_s=result.get('acquisition_time_s'),
                     acquisition_physical_distance_m=result.get('acquisition_physical_distance_m'),
                     acquisition_wheel_distance_m=acq.get('wheel_distance_m'),
                     acquisition_stop=acq.get('exploration_termination'), acquisition_goal_interruptions=len(acq.get('failed_goals', [])),
                     known_free_area_m2=acq.get('known_free_area_m2'), inspection_sim_s=result.get('inspection_time_s'),
                     inspection_wheel_distance_m=result.get('inspection_path_m'), observations=result.get('observations', 0),
                     recoveries=result.get('recoveries', 0), navigation_failure=result.get('navigation_failure', False),
                     inspection_reason=result.get('inspection_reason'), termination=result['termination'],
                     wall_s=run['wall_s'], infrastructure_failure=result['infrastructure_failure'],
                     initializations=run['initializations'], resets_after_start=run['resets_after_start'],
                     interventions=run['interventions'], all_owned_processes_reaped=run['all_owned_processes_reaped'])
        ledger.append(entry)
        truth = truth_pose(out)
        tf = csv_pose(out / 'acquisition/mapper/mapper_tf_pose.csv')
        # This transform is the commissioned initial dock, never a later true pose.
        tf[:, 1:3] += [-2, -2]
        detail = dict(acquisition_localization=error_summary(tf, truth), guards={})
        for stage in ['acquisition', 'inspection']:
            path = out / stage / 'commands.jsonl'
            if path.exists():
                commands = rows(path)
                detail['guards'][stage] = dict(
                    commands=len(commands), modified=sum(r['guard'] is not None for r in commands),
                    reasons=sorted(set(r['guard'] for r in commands if r['guard'] is not None)))
        if (out / 'inspection/task_done.json').exists():
            detail['inspection_localization'] = error_summary(csv_pose(out / 'inspection/estimates.csv'), truth)
            observations = rows(out / 'inspection/private/sensor_truth.jsonl')
            events = rows(out / 'inspection/method.jsonl')
            by_id = {r['id']: r for r in observations}
            target_errors = []
            for event in events:
                if event['event'] != 'observation' or not event.get('observed_asset_map'):
                    continue
                actual = next(a for a in by_id[event['id']]['assets'] if a['actual_id'] == 7)
                estimate = np.array(event['observed_asset_map']['position'])
                target_errors.append(dict(id=event['id'], time=event['sensor_time'],
                                          error_m=float(np.linalg.norm(estimate - actual['asset_pose'][:2]))))
            detail['target_projection_errors'] = target_errors
            detail['wrong_marker_observations'] = [dict(id=e['id'], time=e['sensor_time'],
                ids=e['vision'].get('observed_ids'), decision=e['vision']['identity_decision'])
                for e in events if e['event'] == 'observation' and 8 in e['vision'].get('observed_ids', [])]
            done = load(out / 'inspection/task_done.json')
            final_id = (done.get('final_observation') or {}).get('id')
            final = next((o for o in result['all_observations'] if o['id'] == final_id), None)
            detail['claimed_observation_id'] = final_id if result['success_claim'] else None
            detail['claimed_observation_physically_valid'] = bool(final and final['valid_physical_inspection']) if result['success_claim'] else None
            detail['final_observation'] = final
            handoff = load(out / 'handoff.json')
            t0, t1 = handoff['estimated_state']['time'], result['first_localization_ready']['time']
            selected = truth[(truth[:, 0] >= t0) & (truth[:, 0] <= t1)]
            angles = np.unwrap(selected[:, 3])
            detail['handoff'] = dict(start=t0, localization_ready=t1,
                duration_s=t1-t0, maximum_truth_gap_s=float(np.diff(selected[:, 0]).max()),
                position_motion_max_m=float(np.linalg.norm(selected[:, 1:3] - selected[0, 1:3], axis=1).max()),
                heading_motion_max_deg=float(np.degrees(abs(angles - angles[0]).max())),
                seed_position_error_m=result['handoff_seed_position_error_m'],
                seed_heading_error_deg=result['handoff_seed_heading_error_deg'])
        diagnostics[out.name] = detail
    with (analysis / 'mission_ledger.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
    (analysis / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    protected = load(root / 'preservation_before.json')
    mismatches = [p for p, h in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    frozen = load(root / 'final_freeze.json')
    drift = [p for p, h in frozen['files'].items() if not Path(p).is_file() or sha(Path(p)) != h]
    audit = dict(protected_files=len(protected), protected_mismatches=mismatches,
                 frozen_files=len(frozen['files']), frozen_mismatches=drift,
                 final_freeze_sha256=sha(root / 'final_freeze.json'), attempted_missions=len(ledger),
                 all_attempts_scored=len(ledger) == len(list(root.glob('*/outcome.json'))),
                 all_owned_processes_reaped=all(row['all_owned_processes_reaped'] for row in ledger))
    (analysis / 'integrity.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


def demonstrate(root, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle

    out, destination = root / name, root / 'analysis'
    result, handoff = load(out / 'outcome.json'), load(out / 'handoff.json')
    truth = truth_pose(out)
    estimates = csv_pose(out / 'inspection/estimates.csv')
    acquisition = csv_pose(out / 'acquisition/mapper/mapper_tf_pose.csv'); acquisition[:, 1:3] += [-2, -2]
    t0, t1 = handoff['estimated_state']['time'], result['first_localization_ready']['time']
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for k, label in [(1, 'x'), (2, 'y')]:
        axes[0].plot(truth[:, 0], truth[:, k], label=label+' physical (judge)')
        axes[0].plot(acquisition[:, 0], acquisition[:, k], '--', label=label+' SLAM estimate')
        axes[0].plot(estimates[:, 0], estimates[:, k], ':', label=label+' AMCL estimate')
    axes[0].set(ylabel='Site position (m)', xlim=(t0-5, t1+5))
    axes[0].legend(ncol=3, fontsize=8)
    for label, pose in [('SLAM', acquisition), ('AMCL', estimates)]:
        dx = pose[:, 1] - np.interp(pose[:, 0], truth[:, 0], truth[:, 1])
        dy = pose[:, 2] - np.interp(pose[:, 0], truth[:, 0], truth[:, 2])
        axes[1].plot(pose[:, 0], np.hypot(dx, dy), label=label)
    for ax in axes:
        ax.axvspan(t0, t1, color='grey', alpha=.2); ax.grid(alpha=.2)
    axes[1].set(xlabel='Uninterrupted simulator time (s)', ylabel='Position error (m)'); axes[1].legend()
    fig.suptitle(name + ': stationary estimated-state handoff; no reset')
    fig.tight_layout(); fig.savefig(destination / 'handoff_trace.png', dpi=160); plt.close(fig)

    config = load(out / 'private/evaluation.json')
    sensors = rows(out / 'inspection/private/sensor_truth.jsonl')
    explores = rows(out / 'acquisition/exploration.jsonl')
    decisions = [e for e in explores if e['type'] == 'frontier_decision']
    paths = [e for e in explores if e['type'] == 'path']
    cached = {}
    for d in decisions:
        path = Path(d['map_file'])
        with path.open() as stream:
            head = np.fromstring(stream.readline(), sep=' ')
            grid = np.loadtxt(stream)
        cached[d['map_file']] = (head, grid)
    fig = plt.figure(figsize=(12, 6.4)); ax = fig.add_subplot(121); camera = fig.add_subplot(122)
    writer = FFMpegWriter(fps=10, metadata=dict(title='Continuous sensor-driven map to verified inspection',
        comment='Recorded simulation, evaluator overlays; 10x playback, no interpolated outcomes'), bitrate=1300)
    with writer.saving(fig, str(destination / 'continuous_demo.mp4'), dpi=100):
        for stamp in np.arange(0, truth[-1, 0]+1, 1.):
            ax.clear(); camera.clear()
            recent = [d for d in decisions if d['time'] <= stamp]
            if recent:
                h, grid = cached[recent[-1]['map_file']]
                grey = np.where(grid > 0, .1, np.where(grid < 0, .97, .7))
                ax.imshow(grey, cmap='gray', vmin=0, vmax=1, origin='lower',
                    extent=[h[3]-2-h[2]/2, h[3]-2+(h[0]-.5)*h[2], h[4]-2-h[2]/2, h[4]-2+(h[1]-.5)*h[2]])
            for ox, oy, sx, sy in config['physical_obstacles']:
                ax.add_patch(Rectangle((ox-sx/2, oy-sy/2), sx, sy, fill=False, edgecolor='grey', linewidth=.7))
            before = truth[(truth[:, 0] <= min(stamp, t0))]
            after = truth[(truth[:, 0] >= t0) & (truth[:, 0] <= stamp)]
            ax.plot(before[:, 1], before[:, 2], color='tab:blue', label='physical mapping path')
            if len(after):ax.plot(after[:, 1], after[:, 2], color='tab:orange', label='physical inspection path')
            j = min(np.searchsorted(truth[:, 0], stamp), len(truth)-1); p = truth[j]
            ax.plot(p[1], p[2], 'ko'); ax.arrow(p[1], p[2], .3*np.cos(p[3]), .3*np.sin(p[3]), width=.025, color='k')
            if stamp < t0:
                active = [d for d in paths if d['time'] <= stamp]
                if active:
                    points = np.array(active[-1]['points']) - [2, 2]
                    ax.plot(points[:, 0], points[:, 1], '--', color='tab:green', label='map-selected frontier path')
            else:
                ax.plot(*handoff['amcl_initial_pose'][:2], 'ms', fillstyle='none', label='estimated handoff seed')
            target = next((a for s in sensors for a in s['assets'] if a['actual_id'] == 7), None)
            if target:ax.plot(*target['asset_pose'][:2], '*', color='tab:red', markersize=12, label='target (judge overlay)')
            ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect='equal', xlabel='Site x (m)', ylabel='Site y (m)')
            ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=.15)
            if stamp < t0:phase='Sensor-driven exploration / mapping'
            elif stamp < t1:phase='Stationary map + estimated-pose handoff'
            else:phase='AMCL search, pixel identity, approach and inspection'
            available = [s for s in sensors if s['time'] <= stamp]
            if available:
                s = available[-1]; camera.imshow(plt.imread(out / 'inspection/camera' / s['file']))
                metric = next(o for o in result['all_observations'] if o['id'] == s['id'])
                caption = f"Latest recorded image #{s['id']} at {s['time']:.2f}s\nPixels decoded: {metric['observed_ids']} | reading: {metric['reading_decision']}"
            else:caption='Camera off during mapping; no asset discoveries supplied'
            camera.set_title(caption, fontsize=9); camera.axis('off')
            fig.suptitle(f'{phase}\nSim time {stamp:.1f}s | one world, monotonic odometry | playback 10x', fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, .92]); writer.grab_frame()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    parser.add_argument('--demo', help='Completed mission directory name')
    args = parser.parse_args(); analyze(args.root.resolve())
    if args.demo:demonstrate(args.root.resolve(), args.demo)
