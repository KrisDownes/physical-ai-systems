"""Development tuning and separately frozen, paired arm transfer pilot."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from active_adaptation.arm_transfer import run_episode

OUT = Path('results/arm_transfer')
METHODS = ['nominal', 'online', 'global', 'gp']


def cases(split):
    if split == 'dev':
        poses = [([-.6, -1.25], [.05, -1.55]), ([.1, -1.55], [.7, -1.05]),
                 ([.55, -1.15], [.0, -1.7])]
        changes = [('none', {},), ('gain', {'gain': [.45, 1.]}),
                   ('gain', {'gain': [1., .65]}), ('camera', {'rotation_deg': 10.}),
                   ('camera', {'rotation_deg': -20.})]
        seed = 300
    else:
        poses = [([-.48, -1.35], [.22, -1.65]), ([.22, -1.75], [.72, -1.2]),
                 ([.48, -1.38], [-.12, -1.85]), ([-.12, -1.65], [.42, -1.05])]
        changes = [('none', {}), ('gain', {'gain': [.55, 1.]}),
                   ('gain', {'gain': [1., .75]}), ('camera', {'rotation_deg': -7.}),
                   ('camera', {'rotation_deg': 15.})]
        seed = 700
    result = []
    for i, (start, target) in enumerate(poses):
        for k, (family, disturbance) in enumerate(changes):
            result.append({'id': f'{split}-{i}-{k}', 'family': family,
                           'start_q': start, 'target_q': target, 'seed': seed+i*10+k,
                           'steps': 120, 'change_step': 20, 'noise': .0002, **disturbance})
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metadata():
    source = Path('src/active_adaptation/arm_transfer.py').read_bytes()
    return {'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short', '--', '.'], text=True).strip(),
            'arm_transfer_sha256': hashlib.sha256(source).hexdigest(),
            'python': platform.python_version(), 'numpy': np.__version__}


def development():
    trials = []
    grid = ([('online', {'forgetting': f, 'damping': d}) for f in (.90, .97, .995) for d in (.015, .04)] +
            [('global', {'model_forgetting': f}) for f in (.9, .97, .995)] +
            [('gp', {'model_forgetting': f, 'gp_length': l}) for f in (.9, .97) for l in (.6, 1.2)])
    for method, tuning in grid:
        scores = []
        for case in cases('dev'):
            _, metrics = run_episode(case, method, tuning)
            if metrics['reachable']:
                scores.append(metrics['integrated_normalized_error_s'])
        trials.append({'method': method, 'tuning': tuning, 'mean_error': float(np.mean(scores)),
                       'cases': len(scores)})
        print(method, tuning, round(trials[-1]['mean_error'], 5), flush=True)
    frozen = {'nominal': {}}
    for method in METHODS[1:]:
        frozen[method] = min((t for t in trials if t['method'] == method), key=lambda t: t['mean_error'])['tuning']
    save(OUT / 'development_tuning.json', {'metadata': metadata(), 'cases': cases('dev'), 'trials': trials})
    save(OUT / 'frozen_evaluation.json', {'cases': cases('eval'), 'parameters': frozen,
        'metrics': {'tolerance': .015, 'dwell_steps': 5, 'normalization': .5},
        'note': 'Evaluation frozen after development tuning and before evaluation; no combined disturbances.'})
    print('frozen:', frozen)


def evaluate():
    plan = json.loads((OUT / 'frozen_evaluation.json').read_text())
    rows, traces = [], {}
    for case in plan['cases']:
        for method in METHODS:
            trace, metrics = run_episode(case, method, plan['parameters'][method])
            rows.append({'case': case, 'method': method, 'metrics': metrics})
            traces[f"{case['id']}-{method}"] = trace
    save(OUT / 'evaluation_episodes.json', {'metadata': metadata(), 'episodes': rows})
    save(OUT / 'evaluation_traces.json', traces)
    summary = {}
    for family in ('none', 'gain', 'camera'):
        summary[family] = {}
        for method in METHODS:
            matches = [r['metrics'] for r in rows if r['case']['family'] == family and
                       r['method'] == method and r['metrics']['reachable']]
            summary[family][method] = {'episodes': len(matches),
                'median_integrated_error_s': float(np.median([m['integrated_normalized_error_s'] for m in matches])),
                'successes': sum(m['success'] for m in matches),
                'median_recovery_s_among_successes': (float(np.median([m['recovery_time_s'] for m in matches if m['success']]))
                                                     if any(m['success'] for m in matches) else None),
                'median_peak_error': float(np.median([m['peak_error'] for m in matches])),
                'median_peak_excursion': float(np.median([m['peak_excursion_from_post_change_observation'] for m in matches])),
                'median_action_count': float(np.median([m['action_count'] for m in matches])),
                'median_computation_wall_s': float(np.median([m['computation_wall_s'] for m in matches]))}
    save(OUT / 'evaluation_summary.json', summary)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained', sharey=True)
    for ax, family in zip(axes, ('none', 'gain', 'camera')):
        for method in METHODS:
            ids = [c['id'] for c in plan['cases'] if c['family'] == family]
            e = np.array([[r['error'] for r in traces[f'{i}-{method}'][20:]] for i in ids])
            ax.plot(np.arange(e.shape[1])*.1, np.median(e, axis=0), label=method)
        ax.set_title(family)
        ax.set_xlabel('simulated seconds after change')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('median target error (camera units)')
    axes[-1].legend(fontsize=8)
    fig.savefig(OUT / 'recovery_curves.png', dpi=150)
    failures = [r for r in rows if r['metrics']['reachable'] and not r['metrics']['success']]
    successes = [r for r in rows if r['metrics']['reachable'] and r['metrics']['success']]
    selected = ([max(failures, key=lambda r: r['metrics']['integrated_normalized_error_s'])] if failures else [])
    if successes:
        selected.append(min(successes, key=lambda r: r['metrics']['integrated_normalized_error_s']))
    if selected:
        fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 4), squeeze=False, layout='constrained')
        for ax, record in zip(axes[0], selected):
            name = record['case']['id'] + '-' + record['method']
            trace = traces[name]
            xy = np.array([v['obs'] for v in trace])
            ax.plot(xy[:, 0], xy[:, 1], label='observed tip')
            ax.scatter(*trace[20]['target'], marker='x', color='red', label='final target')
            ax.scatter(*xy[20], marker='o', color='black', s=20, label='change')
            ax.set_title(f"{name}: {'success' if record['metrics']['success'] else 'failure'}")
            ax.set_xlabel('camera x'); ax.set_ylabel('camera y')
            ax.axis('equal'); ax.legend(fontsize=8)
        fig.savefig(OUT / 'representative_episodes.png', dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('dev', 'eval'))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    development() if args.phase == 'dev' else evaluate()
