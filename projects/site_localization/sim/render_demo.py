"""Render measured simulation traces; no reconstruction of unobserved behavior."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);args=p.parse_args();out=args.out
    cfg=json.loads((out/'cell.json').read_text());result=json.loads((out/'outcome.json').read_text())
    rows=[json.loads(x) for x in (out/'judge.jsonl').read_text().splitlines()]
    truth=[r for r in rows if r['type']=='truth'];t=np.array([r['time'] for r in truth])
    xy=np.array([[r['x'],r['y']] for r in truth]);est=np.genfromtxt(out/'estimates.csv',delimiter=',',names=True)
    ex=np.interp(t,est['timestamp'],est['x']);ey=np.interp(t,est['timestamp'],est['y'])
    error=np.hypot(ex-xy[:,0],ey-xy[:,1]);fig,ax=plt.subplots(1,2,figsize=(12,5))
    for x,y,sx,sy in cfg['obstacles']:ax[0].add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,color='.35'))
    goals=np.array(cfg['goals']);ax[0].scatter(goals[:,0],goals[:,1],marker='x',s=60,label='required stops')
    for i,(x,y,_) in enumerate(goals):ax[0].text(x+.08,y+.08,str(i+1))
    ax[0].set(xlim=(-3.4,3.4),ylim=(-3.4,3.4),xlabel='world/map x [m]',ylabel='world/map y [m]')
    ax[0].set_aspect('equal');physical,=ax[0].plot([],[],color='black',label='simulator truth')
    estimated,=ax[0].plot([],[],'--',color='tab:orange',label='AMCL estimate')
    dot,=ax[0].plot([],[],'o',color='tab:blue');ax[0].legend(loc='upper left',fontsize=8)
    ax[1].set(xlim=(0,t[-1]),ylim=(0,max(.15,float(error.max())*1.1)),xlabel='simulation time [s]',ylabel='position error [m]')
    line,=ax[1].plot([],[],color='tab:blue');ax[1].grid(alpha=.3)
    title=fig.suptitle('ROS 2 / Gazebo development mission — simulated robot, analytic prior, noiseless sensors')
    def update(i):
        physical.set_data(xy[:i+1,0],xy[:i+1,1]);estimated.set_data(ex[:i+1],ey[:i+1]);dot.set_data([xy[i,0]],[xy[i,1]])
        line.set_data(t[:i+1],error[:i+1]);title.set_text(f'Simulated inspection mission | t={t[i]:.1f}s | AMCL control; separate truth judge')
        return physical,estimated,dot,line,title
    update(len(t)-1);fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=150)
    frames=np.unique(np.linspace(0,len(t)-1,150).astype(int))
    movie=FuncAnimation(fig,update,frames=frames,interval=100)
    movie.save(out/'demo.mp4',writer=FFMpegWriter(fps=10,bitrate=1500))
    (out/'demo_metrics.json').write_text(json.dumps(dict(interpolated_display_position_rmse_m=float(np.sqrt(np.mean(error**2))),
        mission_success=result['mission_success'],note='display interpolation; outcome.json is the task judge'),indent=2)+'\n')


if __name__=='__main__':main()
