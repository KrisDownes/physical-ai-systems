"""Check negative-elbow nominal reaching before disturbance experiments."""
import json
from pathlib import Path
from active_adaptation.arm_transfer import run_episode

starts = [[-.65, -1.1], [.15, -1.4], [.65, -1.0], [-.25, -1.8]]
targets = [[-.1, -1.35], [.6, -1.1], [.25, -1.6], [.3, -1.25]]
results = []
for i, (start, target) in enumerate(zip(starts, targets)):
    case = {'start_q': start, 'target_q': target, 'seed': 100+i, 'steps': 120,
            'change_step': 20, 'noise': 0.0002}
    _, result = run_episode(case, 'nominal', {})
    results.append({'case': case, 'result': result})
path = Path('results/arm_transfer/nominal_check.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
