"""Plot paired success counts in the frozen integrated pilot."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
root=Path(args.run);s=json.loads((root/'summary.json').read_text())
conds=['none','camera','tool','both','both_no_table'];labels=['No change','Camera','Tool','Camera + tool','Both, no\ntable marks']
methods=['fixed2','calibration','motion','integrated'];names=['Fixed','Calibration only','Motion only','Integrated']
x=np.arange(len(conds));fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
for i,(method,name) in enumerate(zip(methods,names)):
    ax.bar(x+(i-1.5)*.19,[s[c][method]['successes'] for c in conds],.19,label=name)
ax.set_xticks(x,labels);ax.set_yticks([0,1,2]);ax.set_ylim(0,2.5)
ax.set_ylabel('Settled placement successes / 2');ax.set_title('SO-101 integrated calibration pilot')
ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
target=root/'success_counts.png'
if target.exists():raise FileExistsError(target)
fig.savefig(target,dpi=160)
