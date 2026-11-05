"""Post-run physical acquisition evaluation and recorded-map demonstration only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/site_autonomous_plot')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def raster(p):
    with p.open() as f:
        w,h,r,x,y=np.fromstring(f.readline(),sep=' ');a=np.loadtxt(f,dtype=int).reshape(int(h),int(w))
    return a,r,x,y

def evaluate(root,out):
    out.mkdir(parents=True,exist_ok=False);results=[]
    for folder in sorted(root.glob('acquisition__*')):
        if not folder.is_dir() or not(folder/'run.json').exists():continue
        truth=[r for r in rows(folder/'judge.jsonl') if r['type']=='truth']
        t=np.array([r['time'] for r in truth]);xy=np.array([[r['x'],r['y']] for r in truth]);q=np.array([r['q'] for r in truth])
        yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
        cfg=load(folder/'cell.json');done=load(folder/'acquisition_done.json') if (folder/'acquisition_done.json').exists() else {}
        judge=rows(folder/'judge.jsonl');contacts=sum(r['type']=='contacts' and r['count']>0 for r in judge)
        clearance=min(float(np.min(np.hypot(np.maximum(abs(xy[:,0]-ox)-sx/2,0),np.maximum(abs(xy[:,1]-oy)-sy/2,0))-.23)) for ox,oy,sx,sy in cfg['obstacles'])
        dwell=0.;start=None
        for j in range(1,len(t)):
            dt=t[j]-t[j-1];err=abs(np.arctan2(np.sin(yaw[j]),np.cos(yaw[j])))
            valid=(np.linalg.norm(xy[j]-[-2,-2])<=.15 and err<=np.deg2rad(10) and 0<dt<.15 and np.linalg.norm(xy[j]-xy[j-1])/dt<.025 and abs(yaw[j]-yaw[j-1])/dt<.05)
            if valid:
                if start is None:start=t[j-1]
                dwell=t[j]-start
            else:start=None;dwell=0.
        mapped=None
        if (folder/'map_in_site.txt').exists():
            a,res,ox,oy=raster(folder/'map_in_site.txt')
            # Evaluation-only sampling of robot-center feasible physical floor.
            axis=np.arange(-3.2+.025,3.2,.05);xx,yy=np.meshgrid(axis,axis);physical=np.ones_like(xx,dtype=bool)
            for cx,cy,sx,sy in cfg['obstacles']:
                physical &= np.hypot(np.maximum(abs(xx-cx)-sx/2,0),np.maximum(abs(yy-cy)-sy/2,0))>.23
            u=np.rint((xx-ox)/res).astype(int);v=np.rint((yy-oy)/res).astype(int)
            inside=(u>=0)&(v>=0)&(u<a.shape[1])&(v<a.shape[0]);free=np.zeros_like(inside)
            free[inside]=a[v[inside],u[inside]]==-1
            mapped=dict(physical_free_samples=int(physical.sum()),free_classified_samples=int((physical&free).sum()),
                free_fraction=float((physical&free).sum()/physical.sum()),sampling_m=.05,
                definition='Evaluator-only fraction of radius0.23m collision-free centers in declared6.4m cell classified free by transferred map; not a policy coverage signal')
        pose=np.genfromtxt(folder/'mapper/mapper_tf_pose.csv',delimiter=',',names=True)
        pose=np.atleast_1d(pose);supported=(pose['timestamp']>=t[0])&(pose['timestamp']<=t[-1]);pose=pose[supported]
        err=np.hypot(pose['x']-2-np.interp(pose['timestamp'],t,xy[:,0]),pose['y']-2-np.interp(pose['timestamp'],t,xy[:,1]))
        decisions=rows(folder/'exploration.jsonl') if (folder/'exploration.jsonl').exists() else []
        results.append(dict(episode=folder.name,termination=done,contacts=contacts,minimum_clearance_m=clearance,
            physical_dock_end_dwell_s=dwell,physical_dock_return=bool(dwell>=2),
            localization_position_rms_m=float(np.sqrt(np.mean(err**2))),localization_position_max_m=float(err.max()),
            evaluator_map_extent=mapped,frontier_decisions=sum(d['type']=='frontier_decision' for d in decisions),
            goal_interruptions=[d for d in decisions if d['type']=='goal_interrupted'],
            all_attempt_cost_sim_s=float(t[-1]),all_attempt_cost_distance_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())))
    (out/'acquisition_evaluation.json').write_text(json.dumps(results,indent=2)+'\n')
    folder=root/'acquisition__moved__exploration'
    if not (folder/'exploration.jsonl').exists():return
    decisions=[r for r in rows(folder/'exploration.jsonl') if r['type']=='frontier_decision']
    if not decisions:return
    poses=np.genfromtxt(folder/'mapper/mapper_tf_pose.csv',delimiter=',',names=True)
    paths=[r for r in rows(folder/'exploration.jsonl') if r['type']=='path']
    last=load(folder/'acquisition_done.json')['time']
    fig,ax=plt.subplots(figsize=(7,7));writer=FFMpegWriter(fps=5,metadata={'title':'Recorded live-map frontier decisions; no target coordinates'})
    with writer.saving(fig,str(out/'exploration_demo.mp4'),100):
        for stamp in np.linspace(decisions[0]['time'],last,75):
            ax.clear();d=next(d for d in reversed(decisions) if d['time']<=stamp)
            a,r,x,y=raster(Path(d['map_file']));ax.imshow(np.where(a==-1,1,np.where(a==1,0,.55)),origin='lower',extent=[x,x+a.shape[1]*r,y,y+a.shape[0]*r],cmap='gray',vmin=0,vmax=1)
            used=poses[poses['timestamp']<=stamp];ax.plot(used['x'],used['y'],color='#087db6',label='SLAM pose history')
            if len(used):ax.plot(used['x'][-1],used['y'][-1],'o',color='#087db6')
            for c in d['candidates']:
                ax.plot(*c['point'],'o',color='#24a148' if c['eligible'] else '#999999',ms=5)
            available=[p for p in paths if p['time']<=stamp]
            if available:
                p=available[-1];points=np.asarray(p['points']);ax.plot(points[:,0],points[:,1],'--',color='#d45300',label='selected map path');ax.plot(*p['goal'],'x',color='#d45300',ms=10)
            ax.set(xlim=(-1.4,5.4),ylim=(-1.4,5.4),xlabel='SLAM x (m)',ylabel='SLAM y (m)',title=f'Recorded sensor-driven acquisition — t={stamp:.1f}s\n{d["clusters"]} frontier clusters; {d["unreachable_clusters"]} unreachable\nGray is unknown; no asset/world truth shown')
            ax.set_aspect('equal');ax.legend(loc='lower right');fig.tight_layout();writer.grab_frame()
            if stamp==np.linspace(decisions[0]['time'],last,75)[0]:fig.savefig(out/'first_frontier_decision.png',dpi=140)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.runs,a.out)
