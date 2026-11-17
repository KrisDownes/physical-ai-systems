"""Matched idealized placement comparison from recorded author push histories."""
import argparse,functools,gzip,json,sys,time
from pathlib import Path
import imageio.v2 as imageio,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'published_manipulation'))
from run_store import create
from balance import evaluate

def yaw(q):
 q=np.asarray(q)/np.linalg.norm(q);w,x,y,z=q
 return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))
def features(row,kind,other=None):
 # No simulator truth, hidden labels or contact forces are inputs.
 obs=np.asarray(row['observations']);start=obs[0];end=obs[-1];prev=obs[-2]
 a=yaw(start[14:18]);b=yaw(end[14:18]);delta=wrap(b-a)
 if other is not None:
  o=np.asarray(other['observations']);delta=wrap(yaw(o[-1,14:18])-yaw(o[0,14:18]))
 if kind=='start_only':return start.copy()
 if kind=='current_state':return np.r_[end,(end[:3]-prev[:3])/.1,(end[11:14]-prev[11:14])/.1,wrap(b-yaw(prev[14:18]))/.1]
 return np.array([a,delta,start[12]-start[1]])
def fit(rows,kind):
 x=np.stack([features(r,kind) for r in rows]);y=np.array([.6*r['hidden_mass_offset_m_evaluator_training'] for r in rows]);mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.02);z=(x-mean)/scale;design=np.column_stack([np.ones(len(z)),z]);penalty=np.diag([0]+[1.]*len(mean));coef=np.linalg.solve(design.T@design+penalty,design.T@y);return mean,scale,coef
def predict(model,x):
 mean,scale,coef=model;return float(np.clip(np.r_[1.,(x-mean)/scale]@coef,-.06,.06))
@functools.lru_cache(maxsize=None)
def prior_outcome(theta,choice):return evaluate(theta,choice)[0]['success']
def choose_prior(rows):
 candidates=np.linspace(-.06,.06,25);counts=[sum(prior_outcome(r['hidden_mass_offset_m_evaluator_training'],float(v)) for r in rows) for v in candidates];return float(min((v for v,c in zip(candidates,counts) if c==max(counts)),key=abs))
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--data',required=True);p.add_argument('--mode',choices=['development','evaluation'],required=True);a=p.parse_args();train=json.loads(Path(a.train).read_text())['episodes'];data=json.loads(Path(a.data).read_text())['episodes']
 if a.mode=='evaluation':
  assert not ({r['seed'] for r in train}&{r['seed'] for r in data}), 'evaluation seed leakage'
  assert not ({r['hidden_mass_offset_m_evaluator_training'] for r in train}&{r['hidden_mass_offset_m_evaluator_training'] for r in data}), 'evaluation mass leakage'
 cfg={'mode':a.mode,'train':str(Path(a.train).resolve()),'data':str(Path(a.data).resolve()),'models':'ridge alpha 1 with per-fold standardization and 0.02 floor; start-only gets all initial observations, current-state gets all final observations and estimated linear/yaw velocities, history uses start yaw, angular response and initial relative y','training_target':'evaluator-only true COM along rod, 0.6 * hidden mass offset; privileged training labels','negative_control':'history response from next physical case, mismatched initial context; distribution shift possible','candidate_prior_m':np.linspace(-.06,.06,25).tolist(),'placement':'idealized zero-velocity release 0.5mm above 24mm-wide fixed support, 2s at 2ms steps','success':'support contact, tilt < 0.1 rad, speed < 0.05 m/s, angular speed < 0.5 rad/s continuously from 1.5s through 2s','budgets':'fixed priors and start-only: 0 push actions + 2s placement; current/history/mismatched: fixed 20-action 2s push + 2s placement; all within 4s interaction limit; idealized placement reset','development_split':'leave one mass offset out, all starts','evaluation_split':'separate offsets and seeds','no_retries':True}
 out=create('rod_balance_'+a.mode,cfg)
 with gzip.open(out/'rod_source.json.gz','wt') as f:json.dump({p.name:p.read_text() for p in Path(__file__).resolve().parent.glob('*.py')},f)
 t0=time.perf_counter();rows=[];models={kind:fit(train,kind) for kind in ['start_only','current_state','history']};prior=choose_prior(train);foldcache={}
 for i,row in enumerate(data):
  theta=row['hidden_mass_offset_m_evaluator_training'];fold=[r for r in train if r['hidden_mass_offset_m_evaluator_training']!=theta] if a.mode=='development' else train
  if theta not in foldcache:foldcache[theta]=({kind:fit(fold,kind) for kind in models},choose_prior(fold))
  model,fixed=foldcache[theta];other=data[(i+3)%len(data)]
  choices=[('center_prior',0.),('fitted_prior',fixed),('start_only',predict(model['start_only'],features(row,'start_only'))),('current_state',predict(model['current_state'],features(row,'current_state'))),('interaction_history',predict(model['history'],features(row,'history'))),('mismatched_history',predict(model['history'],features(row,'history',other)))]
  for method,choice in choices:
   start=time.perf_counter();result,frames=evaluate(theta,choice,render=(i==0 and method in ('center_prior','interaction_history')));probe=method in ('current_state','interaction_history','mismatched_history');result.update({'case':i,'seed':row['seed'],'hidden_offset_evaluator':theta,'method':method,'support_choice_m':choice,'probe_actions':20 if probe else 0,'probe_sim_s':2. if probe else 0.,'placement_compute_s':time.perf_counter()-start,'contact_samples':row['contact_observed_physics_steps_sampled_at_actions'],'mismatch_source_case':other['case'] if method=='mismatched_history' else None});rows.append(result)
   if frames:imageio.mimsave(out/f'{i:03}_{method}.gif',frames,duration=200,loop=0)
 print({method:sum(r['success'] for r in rows if r['method']==method) for method,_ in choices},'of',len(data),'time',round(time.perf_counter()-t0,2),flush=True)
 serial={k:{'mean':v[0].tolist(),'scale':v[1].tolist(),'coef':v[2].tolist()} for k,v in models.items()}
 (out/'summary.json').write_text(json.dumps({'config':cfg,'prior_m':prior,'models':serial,'results':rows,'wall_s':time.perf_counter()-t0},indent=2));print('saved',out,flush=True)
if __name__=='__main__':main()
