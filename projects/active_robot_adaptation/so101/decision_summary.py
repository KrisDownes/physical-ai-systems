"""Summarize the fixed decision audit without inventing a new test set."""
import argparse,json
from pathlib import Path
import numpy as np
from run_store import create_run
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();data=json.loads((Path(args.run)/'audit.json').read_text());out=create_run('decision_summary',{'audit_run':args.run,'analysis':'one-step candidate comparisons; oracle rollout is evaluator-only'})

def inversions(a,b):
 count=0
 for i in range(len(a)):
  for j in range(i+1,len(a)):
   if (a[i]-a[j])*(b[i]-b[j])<0:count+=1
 return count

def record(r):
 oracle=r['scores_m']['rollout'];n=len(oracle)
 if any(v is None for v in oracle):return {'task':r['task'],'step':r['step'],'missing_oracle_visibility':True}
 def stat(k):
  v=r['scores_m'][k];idx=int(np.argmin(v));measured=np.array([c['visual_next_y_m']-r['visible_feature'][1] for c in r['candidates']]);pred=np.array([c[k+'_predicted_y_m'] for c in r['candidates']]) if k in ('history','current') else np.array([c['action_y_m'] for c in r['candidates']]);return {'winner_m':r['candidates'][idx]['action_y_m'],'oracle_one_step_regret_m':float(oracle[idx]-min(oracle)),'pairwise_ranking_inversions_of_10':inversions(v,oracle),'prediction_rmse_m':float(np.sqrt(np.mean((pred-measured)**2)))}
 return {'task':r['task'],'step':r['step'],'gate_bypasses_planning':r['gate_bypasses_planning'],'recorded_action':r['recorded_action'],'oracle_winner_m':r['winner_action_y_m']['rollout'],'method':{k:stat(k) for k in ('current','history','nominal')},'clone_actual_max_error':r['cloned_vs_actual_next_feature_max_error'],'source_replay_max_error':r['source_replay_max_error'],'candidate_joint_saturations':sum(c['joint_action_saturated'] for c in r['candidates'])}
exact=[]
for t in data['exact_state_counterfactual']:
 a,b=t['low'],t['high'];truth=[abs(x['truth_delta_y_evaluator_m']-y['truth_delta_y_evaluator_m']) for x,y in zip(a['candidates'],b['candidates'])];visible=[abs(x['visual_next_y_m']-y['visual_next_y_m']) for x,y in zip(a['candidates'],b['candidates']) if x['visual_next_y_m'] is not None and y['visual_next_y_m'] is not None]
 exact.append({'step':t['step'],'state_vectors_equal':t['state_vectors_equal'],'low_winner_m':a['winner_action_y_m']['rollout'],'high_winner_m':b['winner_action_y_m']['rollout'],'max_truth_delta_m':max(truth),'max_visible_delta_m':max(visible) if visible else None})
natural=[]
for a,b in zip(data['natural_pair'][0]['stages'],data['natural_pair'][1]['stages']):
 x=np.asarray(a['visible_feature']);y=np.asarray(b['visible_feature'])
 natural.append({'step':a['step'],'xy_difference_m':float(np.linalg.norm(x[:2]-y[:2])),'velocity_difference_m_s':float(np.linalg.norm(x[2:4]-y[2:4])),'fk_tip_difference_m':float(np.linalg.norm(x[4:6]-y[4:6])),'max_joint_difference_rad':float(np.max(np.abs(x[6:12]-y[6:12]))),'low_winner_m':a['winner_action_y_m']['rollout'],'high_winner_m':b['winner_action_y_m']['rollout']})
all_audits=data['recorded_decisions']+[x for t in data['exact_state_counterfactual'] for x in (t['low'],t['high'])]+[x for c in data['natural_pair'] for x in c['stages']]
summary={'recorded':[record(r) for r in data['recorded_decisions']],'natural_pair':natural,'exact_pair':exact,'max_physical_difference_m':max(x['max_truth_delta_m'] for x in exact),'max_visible_difference_m':max(x['max_visible_delta_m'] for x in exact if x['max_visible_delta_m'] is not None),'oracle_winners_match_all':all(x['low_winner_m']==x['high_winner_m'] for x in exact) and all(x['low_winner_m']==x['high_winner_m'] for x in natural),'source_unchanged_all':all(x['source_unchanged'] for x in all_audits),'candidate_numerical_invalid':sum(c['numerical_invalid'] for x in all_audits for c in x['candidates']),'candidate_self_collision_steps':sum(c['self_collision_steps'] for x in all_audits for c in x['candidates']),'candidate_joint_limit_steps':sum(c['joint_limit_steps'] for x in all_audits for c in x['candidates']),'mean_predictor_ms_per_five':float(np.mean([1000*x['all_predictors_compute_s'] for x in all_audits])),'mean_rollout_ms_per_five':float(np.mean([1000*x['cloned_rollouts_wall_s'] for x in all_audits])),'total_wall_s':data['wall_s_before_video']}
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2));print('saved',out)
