"""Summarize fixed pilot results without model selection on test episodes."""
import argparse,gzip,json,statistics
from pathlib import Path
import numpy as np
from run_store import create_run
p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--training',required=True);a=p.parse_args();path=Path(a.evaluation)
if (path/'results.json').exists():rows=json.loads((path/'results.json').read_text())
else:
 with gzip.open(path/'results.json.gz','rt') as f:rows=json.load(f)
training=json.loads((Path(a.training)/'summary.json').read_text());out=create_run('shared_analysis',{'evaluation':a.evaluation,'training':a.training,'analysis':'predeclared aggregate; repeated conventional rows are paired deterministic starts, not independent starts'})
summary={}
for task in ('push','pull'):
 summary[task]={}
 for method in ('conventional','current','history','frozen_history'):
  g=[r for r in rows if r['task']==task and r['method']==method];ok=[r for r in g if 'final_error_evaluator_m' in r]
  summary[task][method]={'successes':sum(r['success'] for r in g),'episodes':len(g),'failures':[r.get('failure') for r in g if not r['success']],
   'median_final_error_m':float(np.median([r['final_error_evaluator_m'] for r in ok])) if ok else None,
   'median_actions':float(np.median([r['actions_including_setup'] for r in ok])) if ok else None,
   'median_movement_rad':float(np.median([r['movement_rad_including_setup'] for r in ok])) if ok else None,
   'median_sim_s':float(np.median([r['sim_s'] for r in ok])) if ok else None,
   'median_wall_s':float(np.median([r['wall_s'] for r in ok])) if ok else None,
   'median_selector_ms_per_step':float(np.median([1000*r['model_selection_compute_s']/len(r['records']) for r in ok if r['records']])) if ok else None,
   'collision_physics_steps':sum(r.get('self_collision_physics_steps',0) for r in g),'limit_physics_steps':sum(r.get('joint_limit_physics_steps',0) for r in g),'numerical_invalid':sum(r.get('numerical_invalid',False) for r in g),
   'active_actions':int(sum(sum(np.linalg.norm(t['action'])>1e-9 for t in r.get('records',[])) for r in g)),'missing':sum(r.get('missing_observations',0) for r in g)}
summary['prediction']={m:{'test_rmse_mm_per_coordinate':[10*np.sqrt(r['test_mse_scaled']) for r in training if r['model'].startswith(m+'_')],'validation_rmse_mm_per_coordinate':[10*np.sqrt(r['validation_mse_scaled']) for r in training if r['model'].startswith(m+'_')]} for m in ('current','history')}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print('saved',out)
