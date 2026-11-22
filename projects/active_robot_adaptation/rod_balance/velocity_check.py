"""Post-hoc comparator audit: add finite-difference joint velocities to B and C."""
import argparse,gzip,json,sys,time
from pathlib import Path
import numpy as np
from iteration import training,hist,outcome,source
from compare import features,predict
from run_store import create

def extra(row):
 o=np.asarray(row['observations']);return (o[-1,3:11]-o[-2,3:11])/.1

def feature(row,kind):return np.r_[features(row,'current_state') if kind=='B' else hist(row),extra(row)]
def fit(rows,kind):
 x=np.stack([feature(r,kind) for r in rows]);y=np.array([.6*r['hidden_mass_offset_m_evaluator_training'] for r in rows]);mean=x.mean(0);scale=np.maximum(x.std(0),.02);z=(x-mean)/scale;X=np.c_[np.ones(len(z)),z];coef=np.linalg.solve(X.T@X+np.diag([0]+[1.]*(X.shape[1]-1)),X.T@y);return mean,scale,coef

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();rows=json.loads(Path(a.data).read_text())['episodes'];out=create('rod_velocity_posthoc',{'reason':'user requested all available velocities; existing B/C omitted final finite-difference joint velocities','status':'exploratory comparator correction, after inspecting frozen outcomes','data':str(Path(a.data).resolve())});source(out);models={k:fit(training(),k) for k in 'BC'};results=[];t=time.perf_counter()
 for r in rows:
  for k in 'BC':
   choice=predict(models[k],feature(r,k));z=outcome(r['hidden_mass_offset_m_evaluator_training'],choice);z.update({'case':r['case'],'method':k,'support_choice_m':choice});results.append(z)
 by={k:[x for x in results if x['method']==k] for k in 'BC'};rng=np.random.default_rng(443);e=np.array([abs(x['placement_error_evaluator_m'])-abs(y['placement_error_evaluator_m']) for x,y in zip(by['C'],by['B'])]);s=np.array([int(x['success'])-int(y['success']) for x,y in zip(by['C'],by['B'])]);inds=rng.integers(0,len(e),(10000,len(e)));summary={'B_success':sum(x['success'] for x in by['B']),'C_success':sum(x['success'] for x in by['C']),'B_mae':float(np.mean([abs(x['placement_error_evaluator_m']) for x in by['B']])), 'C_mae':float(np.mean([abs(x['placement_error_evaluator_m']) for x in by['C']])), 'C_minus_B_error':float(e.mean()),'error_ci':np.quantile(e[inds].mean(1),[.025,.975]).tolist(),'C_only':int(sum(s==1)),'B_only':int(sum(s==-1)),'success_ci':np.quantile(s[inds].mean(1),[.025,.975]).tolist(),'results':results,'wall_s':time.perf_counter()-t};(out/'summary.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='results'},'saved',out,flush=True)
if __name__=='__main__':main()
