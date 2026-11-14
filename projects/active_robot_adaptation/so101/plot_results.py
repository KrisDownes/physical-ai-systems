"""Plot initial SO-101 comparison and labeled follow-up transfer ablation."""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
parser.add_argument('--transfer-run',required=True)
args=parser.parse_args()
out=Path(args.run)
a=json.loads((out/'comparison.json').read_text())['summary']
b=json.loads((Path(args.transfer_run)/'transfer.json').read_text())['summary']
conditions=('none','camera','gain')
labels=('Nominal','Recalibration','Online Jacobian','Global*','Global + GP*')
methods=('nominal','recalibration','online','global','gp')
colors=('#3672a3','#4e9b6a','#c58c39','#9c73b2','#c77391')
x=np.arange(3)
fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
for i,(method,label,color) in enumerate(zip(methods,labels,colors)):
    src=b if method in ('global','gp') else a
    v=[src[c][method]['successes'] for c in conditions]
    ax.bar(x+(i-2)*.15,v,.15,label=label,color=color)
ax.set_xticks(x,['No change','Camera translation','Joint command gain'])
ax.set_yticks(range(5));ax.set_ylim(0,4.8)
ax.set_ylabel('Successful pick-and-place episodes (out of 4)')
ax.set_title('SO-101 simulation: paired pilot')
ax.set_xlabel('* Follow-up ablation reused already-inspected cases.',fontsize=8,labelpad=10)
ax.legend(ncol=3,fontsize=8,loc='upper center')
ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
target=out/'success_comparison.png'
if target.exists(): raise FileExistsError(target)
fig.savefig(target,dpi=160)
