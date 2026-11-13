"""Run a reproducible, paired, development-only camera-change example."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from active_adaptation.evaluate import run


def main():
    start = time.perf_counter()
    config_path = Path('configs/development.json')
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    out = Path('results/development')
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    summary = {}
    for name, adapt in [('fixed', False), ('online', True)]:
        trace, metrics = run(config, adapt)
        summary[name] = metrics
        with (out / f'{name}_trace.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trace[0].keys())
            writer.writeheader()
            writer.writerows(trace)
        ax.plot([row['time_s'] for row in trace], [row['error'] for row in trace], label=name)
    ax.axvline(config['change_step'] * 0.1, color='black', linestyle='--', label='camera change')
    ax.axhline(config['tolerance_camera_units'], color='gray', linestyle=':', label='tolerance')
    ax.set(xlabel='time (s)', ylabel='camera-space target error', title='Development: camera rotation')
    ax.legend()
    fig.savefig(out / 'comparison.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    git = lambda *args: subprocess.check_output(['git', *args], text=True).strip()
    metadata = {'config': config, 'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                'dependency_lock_sha256': hashlib.sha256(Path('requirements.lock.txt').read_bytes()).hexdigest(),
                'git_revision': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
                'numpy_version': np.__version__, 'wall_time_s': time.perf_counter()-start,
                'note': 'One deterministic development example, not a statistical comparison.'}
    (out / 'summary.json').write_text(json.dumps({'metadata': metadata, 'methods': summary}, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
