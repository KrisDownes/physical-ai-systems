"""Bounded launch of the unmodified authors' train.py; CPU only."""
import argparse,csv,json,os,signal,subprocess,time
import psutil
from pathlib import Path
from run_store import create

def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=32);p.add_argument('--max-seconds',type=int,default=90);p.add_argument('--seed',type=int,default=17);a=p.parse_args()
 upstream=Path(os.environ.get('ASID_SRC','/tmp/active-asid-src'));assert (upstream/'train.py').exists()
 cfg={'scope':'unmodified author SAC training on author rod-inertia information reward; CPU; finite-budget feasibility only','steps':a.steps,'max_seconds':a.max_seconds,'seed':a.seed,'workers':1,'eval_workers':1,'learning_starts':500,'train_freq':32,'gradient_steps':1,'author_default_total_timesteps':250000,'paper_downstream_sysid_cem_released':False,'pretrained_policy_released':False}
 out=create('asid_train',cfg);logdir=out/'author_log';cmd=[os.environ.get('ASID_PYTHON','/tmp/active-asid39/bin/python'),str(upstream/'train.py'),'num_workers=1','num_workers_eval=1',f'train.total_timesteps={a.steps}',f'seed={a.seed}','log=stdout','log.eval_interval=100000','log.save_interval=100000',f'log.dir={logdir}'];cfg['command']=cmd
 env=os.environ.copy();env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/active-asid-mpl',WANDB_MODE='disabled',PYTHONPATH=str(upstream))
 start=time.perf_counter();timed=['/usr/bin/time','-v','-o',str(out/'resources.txt')]+cmd
 samples=[]
 with (out/'stdout.log').open('x') as stream:
  proc=subprocess.Popen(timed,cwd=upstream,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  censored=False
  while proc.poll() is None:
   try:
    parent=psutil.Process(proc.pid);family=[parent]+parent.children(recursive=True);rss=0
    for child in family:
     try:rss+=child.memory_info().rss
     except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    samples.append((time.perf_counter()-start,rss,len(family)))
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   if time.perf_counter()-start>=a.max_seconds:
    censored=True;os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    break
   time.sleep(.5)
  code=proc.returncode
 with (out/'process_tree_rss.csv').open('x') as f:
  writer=csv.writer(f,lineterminator='\n');writer.writerow(['wall_s','summed_process_tree_rss_bytes','processes']);writer.writerows(samples)
 result={'exit_code':code,'censored':censored,'wall_s':time.perf_counter()-start,'command':cmd,'max_summed_process_tree_rss_bytes':max((x[1] for x in samples),default=0),'rss_note':'sampled sum of RSS across processes; may double-count shared pages','checkpoints':[str(p.relative_to(out)) for p in out.rglob('*.zip')]}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('saved',out,flush=True)
if __name__=='__main__':main()
