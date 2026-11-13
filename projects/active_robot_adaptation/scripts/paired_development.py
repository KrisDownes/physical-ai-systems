"""Save paired development cases and full traces; not a final comparison."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from active_adaptation.evaluate import run


def main():
    start = time.perf_counter()
    suite_path = Path('configs/paired_development.json')
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    base_path = Path(suite['base_config'])
    base_bytes = base_path.read_bytes()
    base = json.loads(base_bytes)
    output = Path('results/paired_development')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for ax, angle in zip(axes.flat, suite['camera_rotations_deg']):
        config = dict(base, camera_rotation_deg=angle)
        for method in suite['methods']:
            trace, metrics = run(config, method=method)
            row = {'camera_rotation_deg': angle, 'method': method, **metrics}
            rows.append(row)
            with (output / f'rotation_{angle}_{method}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trace[0].keys())
                writer.writeheader()
                writer.writerows(trace)
            ax.plot([r['time_s'] for r in trace], [r['error'] for r in trace], label=method)
        ax.axvline(base['change_step'] * 0.1, color='black', linestyle='--')
        ax.axhline(base['tolerance_camera_units'], color='gray', linestyle=':')
        ax.set_title(f'{angle}° rotation')
        ax.set_xlabel('time (s)')
        ax.set_ylabel('camera-space error')
    axes.flat[-1].axis('off')
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=120)
    plt.close(fig)
    git = lambda *args: subprocess.check_output(['git', *args], text=True).strip()
    result = {'metadata': {'suite': suite, 'base_config': base,
              'suite_sha256': hashlib.sha256(suite_bytes).hexdigest(),
              'base_sha256': hashlib.sha256(base_bytes).hexdigest(),
              'lock_sha256': hashlib.sha256(Path('requirements.lock.txt').read_bytes()).hexdigest(),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('src/active_adaptation/plant.py'), Path('src/active_adaptation/observations.py'), Path('src/active_adaptation/controllers.py'), Path('src/active_adaptation/evaluate.py'), Path('scripts/paired_development.py')]},
              'git_revision': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain')),
              'python': platform.python_version(), 'numpy': np.__version__,
              'wall_time_s': time.perf_counter()-start,
              'limitations': 'Development sweep; one initial state and target, no noise; no tuned or held-out comparison.'},
              'cases': rows}
    (output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in rows:
        print(f"{row['camera_rotation_deg']:>4}° {row['method']:>6} error={row['integrated_normalized_error_s']:.3f} recovery={row['recovery_time_s']}")


if __name__ == '__main__':
    main()
