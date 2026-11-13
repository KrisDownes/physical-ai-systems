"""Post-evaluation diagnostic: paired negative and positive elbow poses."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from active_adaptation.arm_transfer import run_episode
from active_adaptation.plant import PlanarPlant

out = Path('results/arm_transfer')
plan = json.loads((out / 'frozen_evaluation.json').read_text())
base_cases = [c for c in plan['cases'] if c['family'] == 'none']
records = []
plant = PlanarPlant()
for case in base_cases:
    xy = plant.reset(case['start_q'])
    q2 = -case['start_q'][1]
    q1 = np.arctan2(xy[1], xy[0]) - np.arctan2(.2*np.sin(q2), .3+.2*np.cos(q2))
    assert np.allclose(plant.reset([q1, q2]), xy)
    for branch, initial in [('negative', case['start_q']), ('positive', [q1, q2])]:
        pre_target = ((np.asarray(case['start_q']) + np.asarray(case['target_q'])) / 2).tolist()
        trial = {**case, 'start_q': initial, 'pre_target_q': pre_target}
        for method, tuning in plan['parameters'].items():
            trace, metrics = run_episode(trial, method, tuning)
            records.append({'case_id': case['id'], 'branch': branch, 'method': method,
                            'metrics': metrics, 'trace': trace})
(out / 'branch_stress.json').write_text(json.dumps(records, indent=2) + '\n')
for branch in ('negative', 'positive'):
    for method in plan['parameters']:
        rows = [r['metrics'] for r in records if r['branch'] == branch and r['method'] == method]
        print(branch, method, 'success', sum(r['success'] for r in rows), '/', len(rows),
              'median error', round(float(np.median([r['integrated_normalized_error_s'] for r in rows])), 4))
failure = next((r for r in records if r['branch'] == 'positive' and not r['metrics']['success']), None)
if failure:
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for method in plan['parameters']:
        r = next(r for r in records if r['case_id'] == failure['case_id'] and r['branch'] == 'positive' and r['method'] == method)
        ax.plot(np.arange(100)*.1, [x['error'] for x in r['trace'][20:]], label=method)
    ax.axhline(.015, color='black', linestyle='--', label='tolerance')
    ax.set(xlabel='simulated seconds after target switch', ylabel='target error (camera units)',
           title=f"Positive-elbow stress: {failure['case_id']}")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(out / 'branch_stress.png', dpi=150)
