"""Plot retained physical trajectory and analytic camera observations only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main(run,out):
    if out.exists():raise FileExistsError(out)
    truth=[json.loads(s) for s in (run/'judge.jsonl').read_text().splitlines() if json.loads(s)['type']=='truth']
    captures=[json.loads(s) for s in (run/'sensor_truth.jsonl').read_text().splitlines()]
    trace=np.array([[r['x'],r['y']] for r in truth])
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),gridspec_kw={'width_ratios':[1.25,1,1]})
    ax=axes[0];ax.plot(trace[:,0],trace[:,1],color='#136f63',label='Measured Gazebo trajectory')
    ax.add_patch(Rectangle((-4.265,-.13),.03,.26,color='#777777',label='Overhead panel'))
    ax.plot([-3.5,-3.5],[-.14,.14],color='#5a3684',lw=4,label='Commissioned fixture')
    for i,c in enumerate(captures):
        x,y,yaw=c['pose'];color=['#bd4f1c','#136f63'][i]
        ax.scatter(x,y,c=color,s=55,zorder=3)
        ax.plot([x,-3.5],[y,0],color=color,ls='--',lw=1)
        ax.annotate(['First capture','Second capture'][i],(x,y),xytext=(-4.94,-.35 if i==0 else .62),fontsize=9)
    ax.set(xlim=(-6.2,-3.25),ylim=(-.55,.85),xlabel='Mission x [m]',ylabel='Mission y [m]',title='One bounded observation move')
    ax.set_aspect('equal');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
    for i,c in enumerate(captures):
        ax=axes[i+1];ax.imshow(plt.imread(run/'camera'/c['file']),cmap='gray',vmin=0,vmax=1)
        ax.set(xlim=(650,1450),ylim=(850,230))
        ax.set_title(['Arrival: abstain','After move: accepted reading'][i],fontsize=11)
        ax.set_xlabel(f"Capture at {c['time']:.3f} simulated seconds")
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Development integration: synthetic camera, fixed reader, actual simulated motion',fontsize=12)
    fig.tight_layout();fig.savefig(out,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.run,a.out)
