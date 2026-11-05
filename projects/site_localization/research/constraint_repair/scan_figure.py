"""Evaluator-only drawing of recorded lidar scans around the first false link."""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import truth_pose

base=ROOT/'results/connected_layout_20261025'
out=ROOT/'results/constraint_repair_20261026/visible_scan_alias.png'
nodes={int(r['id']):r for r in csv.DictReader(open(base/'analysis/visible_graph_nodes.csv'))}
edge=next(r for r in csv.DictReader(open(base/'analysis/visible_graph_edges.csv')) if (r['source'],r['target'])==('91','314'))
times=[float(nodes[i]['time']) for i in (91,314)]
frames={}
with (base/'mission_visible/acquisition/mapper/sensor_frames.txt').open() as f:
    next(f)
    for line in f:
        s=line.split();t=float(s[0])
        for target in times:
            if abs(t-target)<1e-5:frames[target]=s
        if len(frames)==2:break
truth=truth_pose(base/'mission_visible')
def physical(t):return np.array([np.interp(t,truth[:,0],truth[:,k]) for k in (1,2,3)])
def hits(row,pose):
    n=int(row[4]);ranges=np.array([float(row[5+2*j]) for j in range(n)]);angles=np.array([float(row[6+2*j]) for j in range(n)])
    ok=np.isfinite(ranges)&(ranges>.05)&(ranges<7.9)
    return np.stack((pose[0]+ranges[ok]*np.cos(angles[ok]+pose[2]),pose[1]+ranges[ok]*np.sin(angles[ok]+pose[2])),axis=1)
a,b=map(physical,times);d=np.array([float(edge[k]) for k in ('dx','dy','dyaw')]);c,s=math.cos(a[2]),math.sin(a[2]);wrong=np.array([a[0]+c*d[0]-s*d[1],a[1]+s*d[0]+c*d[1],a[2]+d[2]])
fig,ax=plt.subplots(figsize=(8,7))
for label,t,pose,color,mark in [('earlier scan, physical',times[0],a,'#1d5ca8','o'),('later scan, physical',times[1],b,'#e37126','o'),('later scan, false link',times[1],wrong,'#ad1c43','x')]:
    h=hits(frames[t],pose);ax.scatter(h[:,0],h[:,1],s=5,alpha=.35,color=color,label=label+' returns')
    ax.scatter([pose[0]],[pose[1]],color=color,s=110,marker=mark,label=label+' pose')
ax.plot([b[0],wrong[0]],[b[1],wrong[1]],'k--',lw=1,label='physical vs inferred placement')
ax.set(xlabel='physical x (m)',ylabel='physical y (m)',title='Visible case: stored 91→314 link places the later scan in the wrong room')
ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='upper right');fig.tight_layout();fig.savefig(out,dpi=170)
print(out, 'source',a,'target',b,'false_pose',wrong,'recorded_ranges',len(frames[times[0]]))
