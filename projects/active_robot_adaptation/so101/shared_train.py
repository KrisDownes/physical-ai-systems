"""Fixed-window observable dynamics baseline; NumPy CPU with checked gradients."""
import argparse,gzip,hashlib,json,platform,resource,subprocess,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
SCALE=np.array([.2,.2,.1,.1,.2,.2,1,1,1,1,1,1,1],np.float32)
def vector(obs,action):return np.r_[np.asarray(obs,dtype=np.float32)/SCALE,np.asarray(action,dtype=np.float32)/.01]
def build(episodes,split,history):
    xs=[];ys=[]
    for ep in episodes:
        if ep['split']!=split or ep.get('numerical_invalid',False):continue
        context=[]
        for t in ep['transitions']:
            v=vector(t['observation'],t['action']);a=t['observation'];b=t['next_observation']
            if a[-1] and b[-1]:
                padded=([np.zeros(15,np.float32)]*3+context)[-3:]
                xs.append(np.r_[v,*padded] if history else v);ys.append((np.asarray(b[:2])-a[:2])/.01)
            context.append(v)
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32)
def predict(w,x):return np.tanh(x@w[0]+w[1])@w[2]+w[3]
def gradient(w,x,y):
    h=np.tanh(x@w[0]+w[1]);err=(h@w[2]+w[3]-y);d=2*err/err.size
    dh=(d@w[2].T)*(1-h*h)
    return float(np.mean(err**2)),[x.T@dh,dh.sum(0),h.T@d,d.sum(0)]
def run():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--updates',type=int,default=400);p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);args=p.parse_args();data=Path(args.data)
 files=sorted(data.glob('episode_*.json'))
 eps=[json.loads(f.read_text()) for f in files] if files else json.loads((data/'episodes.json').read_text())
 assert {'push','pull'}<=set(e['task'] for e in eps if e['split']=='train'), 'need both tasks before training'
 revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');out=Path('results/so101_runs')/f'{stamp}_{revision[:12]}_shared_train_{uuid.uuid4().hex[:8]}';out.mkdir()
 source={str(f):f.read_text() for f in sorted(Path('so101').glob('*.py'))+sorted(Path('so101').glob('*.json'))+[Path('so101/scene.xml')]}
 with gzip.open(out/'source_snapshot.json.gz','wt') as f:json.dump(source,f)
 meta={'revision':revision,'source_status':subprocess.check_output(['git','status','--short','--','so101'],text=True).strip(),'data_run':str(data),'episodes':len(eps),'args':vars(args),'python':platform.python_version(),'numpy':np.__version__,'implementation':'NumPy dense tanh network with explicit backpropagation and Adam; CPU','source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()}}
 (out/'metadata.json').write_text(json.dumps(meta,indent=2));summary=[]
 for hist in (False,True):
  x,y=build(eps,'train',hist);vx,vy=build(eps,'validation',hist);tx,ty=build(eps,'test',hist)
  if not len(x):raise RuntimeError('no valid visible transitions')
  for seed in (args.seeds[:1] if args.smoke else args.seeds):
   rng=np.random.default_rng(seed);w=[rng.normal(0,.08,(x.shape[1],32)).astype(np.float32),np.zeros(32,np.float32),rng.normal(0,.08,(32,2)).astype(np.float32),np.zeros(2,np.float32)]
   # A finite-difference check on the exact model/loss before any optimizer updates.
   xx=x[:min(8,len(x))];yy=y[:len(xx)];_,g=gradient(w,xx,yy);delta=1e-3;old=w[0][0,0];w[0][0,0]=old+delta;plus=np.mean((predict(w,xx)-yy)**2);w[0][0,0]=old-delta;minus=np.mean((predict(w,xx)-yy)**2);w[0][0,0]=old
   grad_error=abs(float((plus-minus)/(2*delta)-g[0][0,0]));assert grad_error<1e-3,grad_error
   m=[np.zeros_like(a) for a in w];v=[np.zeros_like(a) for a in w];start=time.perf_counter();losses=[];updates=60 if args.smoke else args.updates
   # Smoke intentionally overfits eight samples to validate plumbing, never used for evaluation.
   train_x,train_y=(xx,yy) if args.smoke else (x,y)
   for i in range(1,updates+1):
    idx=rng.integers(0,len(train_x),min(64,len(train_x)));loss,g=gradient(w,train_x[idx],train_y[idx]);losses.append(loss)
    for j in range(4):
     m[j]=.9*m[j]+.1*g[j];v[j]=.999*v[j]+.001*g[j]**2;w[j]-=.002*(m[j]/(1-.9**i))/(np.sqrt(v[j]/(1-.999**i))+1e-8)
   elapsed=time.perf_counter()-start
   def mse(xx,yy):return None if not len(xx) else float(np.mean((predict(w,xx)-yy)**2))
   name=('history' if hist else 'current')+f'_seed{seed}';np.savez_compressed(out/(name+'.npz'),**{f'p{i}':a for i,a in enumerate(w)})
   row={'model':name,'train_samples':len(x),'validation_samples':len(vx),'test_samples':len(tx),'params':sum(a.size for a in w),'updates':updates,'batch':min(64,len(train_x)),'elapsed_s':elapsed,'examples_per_second':updates*min(64,len(train_x))/elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gradient_abs_error':grad_error,'loss_first':losses[0],'loss_last':losses[-1],'train_mse_scaled':mse(x,y),'validation_mse_scaled':mse(vx,vy),'test_mse_scaled':mse(tx,ty)};summary.append(row);print(row,flush=True)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print('saved',out)
if __name__=='__main__':run()
