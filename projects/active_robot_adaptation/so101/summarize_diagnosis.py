"""Split controller observations from evaluator truth and summarize the causal trace."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
root=Path(args.run); rows=json.loads((root/'diagnosis.json').read_text())
controller={};evaluator={};summary={}
for r in rows:
    name=r['name']; trace=r['trace']
    controller[name]={'localization':r.get('localization'),
      'trace':[{k:v for k,v in t.items() if not k.endswith('_evaluator')} for t in trace]}
    evaluator[name]={'initial_object_xyz':r.get('evaluator_initial_object_xyz'),
      'initial_site_xyz':r.get('evaluator_initial_gripper_site_xyz'),
      'truth_trace':[{k:v for k,v in t.items() if k=='time_s' or k.endswith('_evaluator')} for t in trace],
      'evaluation':r.get('evaluation'),'physics_collision_steps':r['collision_steps'],
      'physics_collision_pairs':r['collision_pairs']}
    truth=np.asarray(r['evaluator_initial_object_xyz'][:2]); estimate=np.asarray(r['localization']['object_estimate_xyz'][:2])
    reach=[t for t in trace if t.get('stage')=='reach']
    adapt=[t['adaptation'] for t in trace if t.get('stage') in ('approach','reach') and t.get('adaptation',{}).get('accepted')]
    key='input_u' if r['method']=='online' else 'input_predicted_xyz'
    x=np.asarray([a[key] for a in adapt]) if adapt else np.empty((0,0))
    summary[name]={'success':r['success'],'failure':r['failure'],
      'object_estimate_bias_mm':((estimate-truth)*1000).tolist(),
      'last_reach_site_error_mm':float(np.linalg.norm(np.asarray(reach[-1]['site_xyz_evaluator'])-reach[-1]['target_xyz'])*1000) if reach else None,
      'pre_grasp_accepted_updates':len(adapt),
      'pre_grasp_input_singular_values':np.linalg.svd(x,compute_uv=False).tolist() if adapt else [],
      'pre_grasp_input_rank':int(np.linalg.matrix_rank(x)) if adapt else 0,
      'grasp_contact':r.get('evaluation',{}).get('grasp_contact'),
      'lifted':r.get('evaluation',{}).get('lifted'),
      'physics_collision_steps':r['collision_steps']}
for filename,data in [('controller_log.json',controller),('evaluator_log.json',evaluator),('causal_summary.json',summary)]:
    with (root/filename).open('x') as f:json.dump(data,f,indent=2)
print(json.dumps(summary,indent=2))
