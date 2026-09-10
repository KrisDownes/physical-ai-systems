"""Sequential supervisor-only comparison; no navigation decisions or driver feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(output, records):
    rows = []
    for record in records:
        folder = Path(record['path'])/'episode-01'
        manifest, evaluation = read(folder/'manifest.json'), read(folder/'evaluation.json')
        latency = evaluation.get('latency', {}).get('distributions', {})
        coverage = evaluation.get('final_coverage') or {}
        timeline = read(folder/'action_timeline.json') or []
        movements = [r for r in timeline if r['action'] in ('drive','turn')]
        rows.append({**record, 'actual_model':manifest.get('model_metadata', {}).get('actual_model'),
            'known_grid_percent':coverage.get('known_map_percent'),
            'unresolved_components':coverage.get('unresolved_components'),
            'coverage_complete':evaluation.get('eventual_coverage_completion'),
            'benchmark_pass':evaluation.get('original_benchmark_pass'),
            'termination':evaluation.get('termination', 'pending'),
            'decision_median_s':latency.get('observation_ready_to_request_s', {}).get('median'),
            'dispatch_median_s':latency.get('ros_transport_s', {}).get('median'),
            'movement_requests':len(movements),
            'movement_aborts':sum(any(e.get('kind')=='executor_status' and e.get('state')=='aborted'
                                      for e in r.get('events', [])) for r in movements),
            'sim_time_at_final_map_s':coverage.get('sim_s'),
            'error':manifest.get('error'), 'mission_wall_s':manifest.get('mission_wall_s')})
    (output/'comparison.json').write_text(json.dumps(rows, indent=2))
    lines = ['# Full-map exploration comparison', '',
        'Single-trial pilot; incomplete runs and budgets are reported, not treated as reliable success rates.',
        'Known-grid percentage uses each SLAM rectangular map, so percentages across different maps are not normalized reachable-area coverage.', '',
        '| Condition | Model | Known grid % | Frontiers ≥5 cells | Completion | Benchmark | Termination | Ready→request median s | ROS median s |',
        '|---|---|---:|---:|---|---|---|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(key)) for key in (
            'name','actual_model','known_grid_percent','unresolved_components','coverage_complete',
            'benchmark_pass','termination','decision_median_s','dispatch_median_s')) + ' |')
    lines += ['', 'Exact paths, settings, errors and wall durations: `comparison.json`.',
              'Ready→request includes harness, network and model processing; it is not pure inference latency.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'config/exploration_matrix.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-baseline', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = read(args.config)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    supervisor = {'pid':os.getpid(), 'started_unix_s':time.time(), 'output':str(args.output)}
    (args.output/'supervisor.json').write_text(json.dumps(supervisor, indent=2))
    (args.output/'matrix.json').write_text(json.dumps(config, indent=2))
    records = []
    failed = False
    for condition in config['conditions']:
        target = (args.reuse_baseline.resolve() if condition['name']=='baseline' and args.reuse_baseline
                  else args.output/condition['name'])
        command = [str(ROOT/'scripts/run_rover_experiment'), '--task', 'explore-map', '--episodes', '1',
            '--world', condition['world'], '--scenario', condition['scenario'], '--model', condition['model'],
            '--timeout', str(config['timeout_s']), '--action-budget', str(config['action_budget']),
            '--gui', '--live-stats', '--output', str(target)]
        record = {**condition, 'path':str(target), 'command':command}
        records.append(record)
        summarize(args.output, records)
        if condition['name']=='baseline' and args.reuse_baseline:
            manifest = read(target/'episode-01/manifest.json')
            expected = {k:condition[k] for k in ('world','scenario')}
            if (not manifest.get('ended_unix_s') or manifest.get('model_requested') != condition['model']
                or manifest.get('task_type') != 'explore-map'
                or any(manifest.get(k)!=v for k,v in expected.items())
                or manifest.get('wall_budget_s') != config['timeout_s']
                or manifest.get('action_budget') != config['action_budget']):
                raise RuntimeError('Baseline is incomplete or does not match the matrix settings')
            if manifest.get('error'):
                raise RuntimeError('Baseline has an infrastructure error; resolve it before comparisons')
            continue
        if not args.execute:
            continue
        print('Starting comparison', condition['name'], flush=True)
        with (args.output/(condition['name']+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record['exit_code'] = result.returncode
        summarize(args.output, records)
        print('Finished', condition['name'], 'exit', result.returncode, flush=True)
        if result.returncode:
            print('Stopping matrix after execution/infrastructure failure; see logs.', flush=True)
            failed = True
            break
    summarize(args.output, records)
    supervisor['ended_unix_s'] = time.time()
    (args.output/'supervisor.json').write_text(json.dumps(supervisor, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
