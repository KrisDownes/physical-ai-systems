"""Plot fresh held-out camera-condition success rates."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
root=Path(args.run);s=json.loads((root/'summary.json').read_text())
conditions=['no_change','visible','temporary','unavailable']
labels=['No change\n(n=2)','References\nvisible (n=4)','References\ntemporary (n=4)','References\nunavailable (n=4)']
methods=['nominal','online','recalibration','selfcal']
names=['Fixed nominal','Online Jacobian','Table recalibration','Gripper-marker self-cal']
x=np.arange(4)
fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
for i,(method,name) in enumerate(zip(methods,names)):
    v=[s[c][method]['successes']/s[c][method]['trials'] for c in conditions]
    ax.bar(x+(i-1.5)*.19,v,.19,label=name)
ax.set_xticks(x,labels);ax.set_ylim(0,1.23);ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
ax.set_ylabel('Settled placement success');ax.set_title('Fresh SO-101 camera-translation pilot')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(ncol=2,fontsize=8,loc='upper center')
target=root/'success_rates.png'
if target.exists():raise FileExistsError(target)
fig.savefig(target,dpi=160)
