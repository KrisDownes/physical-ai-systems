"""Export completed comparison measurements; not an online controller."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/rover-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('folder',type=Path)
args=parser.parse_args()
rows=json.loads((args.folder/'comparison.json').read_text())
rows=[r for r in rows if r.get('known_grid_percent') is not None]
fig,axs=plt.subplots(2,2,figsize=(12,8))
for ax,key,label in zip(axs.flat,
    ('known_grid_percent','unresolved_components','decision_median_s','dispatch_median_s'),
    ('Known rectangular grid (%) — not reachable-area coverage','Unresolved frontier components ≥5 cells',
     'Observation-ready → request median (s)','ROS dispatch median (ms)')):
    values=[r[key] if r.get(key) is not None else float('nan') for r in rows]
    if key=='dispatch_median_s':values=[v*1000 for v in values]
    bars=ax.bar([r['name'] for r in rows],values,color='#4677a9')
    ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set_title(label,fontsize=10);ax.tick_params(axis='x',rotation=15)
    ax.margins(y=.2)
fig.suptitle('Camera-only exploration: single-trial pilot, not a reliability estimate')
fig.tight_layout()
fig.savefig(args.folder/'comparison.png',dpi=150)
