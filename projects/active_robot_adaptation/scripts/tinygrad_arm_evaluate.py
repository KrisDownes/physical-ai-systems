"""Evaluate frozen contextual-bandit policies with and without digital correction."""
import json
import time
from pathlib import Path
import numpy as np
from active_adaptation.arm_transfer import desired_step
from active_adaptation.observations import Camera
from active_adaptation.plant import PlanarPlant
from active_adaptation.evaluate import reachable

OUT = Path('results/arm_transfer')
plan = json.loads((OUT / 'frozen_evaluation.json').read_text())
records = []
for seed in (1, 2, 3):
    with np.load(OUT / 'tinygrad' / f'policy_seed_{seed}.npz') as loaded:
        w1, b1, w2, b2 = [loaded[f'p{i}'] for i in range(4)]
    for case in plan['cases']:
        for adapted in (False, True):
            plant, pre = PlanarPlant(), Camera()
            angle = np.deg2rad(case.get('rotation_deg', 0.))
            post = Camera(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
            target = pre.observe(plant.reset(case['target_q']))
            pre_target = pre.observe(plant.reset((np.asarray(case['start_q']) + case['target_q'])/2))
            plant.reset(case['start_q'])
            noise = np.random.default_rng(case['seed']).normal(0, case['noise'], (case['steps']+1, 2))
            b, p, previous = np.eye(2), np.eye(2)*1000., None
            trace, computation = [], 0.
            for step in range(case['steps']):
                changed = step >= case['change_step']
                obs = (post if changed else pre).observe(plant.tip_m()) + noise[step]
                goal = target if changed else pre_target
                start = time.perf_counter()
                if adapted and previous is not None:
                    old_obs, old_desired = previous
                    if np.linalg.norm(old_desired) > .001:
                        delta = obs-old_obs
                        pz = p @ old_desired
                        k = pz/(.9 + old_desired @ pz)
                        b += np.outer(delta-b@old_desired, k)
                        p = (p-np.outer(k, old_desired@p))/.9
                desired = desired_step(obs, goal)
                corrected = np.linalg.solve(b.T@b + .02**2*np.eye(2), b.T@desired) if adapted else desired
                norm = np.linalg.norm(corrected)
                corrected *= min(1., .025/max(norm, 1e-12))
                feature = np.r_[obs/.5, corrected/.025]
                u = .1 * np.tanh(np.tanh(feature@w1+b1)@w2+b2)
                computation += time.perf_counter()-start
                previous = (obs.copy(), corrected.copy())
                gain = np.asarray(case.get('gain', [1., 1.])) if changed else np.ones(2)
                plant.step(u*gain)
                trace.append({'step': step, 'obs': obs.tolist(), 'target': goal.tolist(),
                              'command_rad': u.tolist(), 'error': float(np.linalg.norm(goal-obs))})
            errors = np.array([r['error'] for r in trace[case['change_step']:]])
            within = np.convolve((errors <= .015).astype(int), np.ones(5, int), 'valid')
            hits = np.flatnonzero(within == 5)
            pos = np.array([r['obs'] for r in trace[case['change_step']:]])
            metrics = {'integrated_normalized_error_s': float(errors.sum()*.1/.5),
                       'success': len(hits)>0,
                       'recovery_time_s': float((hits[0]+4)*.1) if len(hits) else None,
                       'peak_error': float(errors.max()),
                       'peak_excursion': float(np.linalg.norm(pos-pos[0], axis=1).max()),
                       'action_count': case['steps'], 'computation_wall_s': computation,
                       'reachable': bool(reachable(target, post))}
            records.append({'training_seed': seed, 'case': case, 'adapted': adapted,
                            'metrics': metrics, 'trace': trace})
summary = {}
for seed in (1, 2, 3):
    summary[str(seed)] = {}
    for adapted in (False, True):
        rows = [r['metrics'] for r in records if r['training_seed']==seed and r['adapted']==adapted and r['metrics']['reachable']]
        summary[str(seed)][str(adapted)] = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
            'median_integrated_error_s': float(np.median([r['integrated_normalized_error_s'] for r in rows]))}
        summary[str(seed)][str(adapted)]['families'] = {}
        for family in ('none', 'gain', 'camera'):
            fam = [r['metrics'] for r in records if r['training_seed']==seed and r['adapted']==adapted
                   and r['case']['family']==family and r['metrics']['reachable']]
            summary[str(seed)][str(adapted)]['families'][family] = {
                'episodes': len(fam), 'successes': sum(r['success'] for r in fam),
                'median_integrated_error_s': float(np.median([r['integrated_normalized_error_s'] for r in fam]))}
(OUT / 'tinygrad' / 'evaluation.json').write_text(json.dumps({'episodes': records, 'summary': summary}, indent=2)+'\n')
print(json.dumps(summary, indent=2))
