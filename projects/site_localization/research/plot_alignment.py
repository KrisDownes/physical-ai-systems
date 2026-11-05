"""Evaluation-only overlay: simulator geometry never enters registration."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from register_maps import occupied_points,transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('survey',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads((a.survey/'cell.json').read_text())
    points,_=occupied_points(a.survey/'mapper/map.txt')
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True,sharey=True)
    for ax,name,label in zip(axes,['alignment.json','alignment_stable_left.json'],['Global robust ICP','Extra unchanged-region information + robust ICP']):
        pose=json.loads((a.survey/'mapper'/name).read_text())['target_from_source'];mapped=transform(points,pose)
        for box in cfg['obstacles']:
            x,y,sx,sy=box;ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,facecolor='0.7',edgecolor='0.4'))
        ax.scatter(mapped[:,0],mapped[:,1],s=2,label='Acquired occupied cells after alignment')
        goals=cfg['goals'][:4];ax.scatter([g[0] for g in goals],[g[1] for g in goals],marker='x',color='black',s=50,label='Fixed physical targets')
        ax.set(title=label,ylabel='Mission-frame y (m)',aspect='equal',xlim=(-8.6,8.6),ylim=(-3.6,3.6))
        ax.legend(loc='upper center',fontsize=8);ax.grid(alpha=.15)
    axes[-1].set_xlabel('Mission-frame x (m)')
    fig.suptitle('Diagnostic only: gray geometry is simulator truth, not a registration input')
    fig.tight_layout();fig.savefig(a.out,dpi=160)
    print(a.out)


if __name__=='__main__':main()
