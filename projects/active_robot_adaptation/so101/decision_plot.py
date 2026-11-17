"""Draw two fixed audit comparisons without overwriting existing artifacts."""
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);out=Path(sys.argv[2]);data=json.loads((run/'audit.json').read_text());s=['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="410">','<rect width="820" height="410" fill="white"/>','<text x="30" y="26" font-size="17">Prediction-to-action audit: fixed one-step candidates</text>']
colors=['#2360ac','#cc6e00']
def panel(x0,title,series,labels,ylow,yhigh):
 s.append(f'<text x="{x0}" y="60" font-size="14">{title}</text>')
 s.append(f'<path d="M{x0} 320 H{x0+335} M{x0} 95 V320" stroke="black" fill="none"/>')
 for n in range(5):
  x=x0+n*75;s.append(f'<text x="{x-12}" y="340" font-size="10">{[-12,-6,0,6,12][n]}</text>')
 for j in range(5):
  val=ylow+(yhigh-ylow)*j/4;y=320-j*225/4;s.append(f'<text x="{x0-34}" y="{y+4}" font-size="10">{val:.1f}</text><path d="M{x0} {y} H{x0+335}" stroke="#ddd"/>')
 for i,ys in enumerate(series):
  pts=' '.join(f'{x0+n*75},{320-(v-ylow)/(yhigh-ylow)*225:.1f}' for n,v in enumerate(ys))
  s.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i]}" stroke-width="2"/><text x="{x0+5}" y="{77+i*14}" fill="{colors[i]}" font-size="11">{labels[i]}</text>')
 s.append(f'<text x="{x0+55}" y="368" font-size="11">candidate y displacement (mm)</text>')
r=next(x for x in data['recorded_decisions'] if x['task']=='push' and x['step']==1)
panel(65,'Push step 1: one-step score (mm)',[[v*1000 for v in r['scores_m'][k]] for k in ('history','rollout')],['history prediction','cloned simulator'],0,25)
e=data['exact_state_counterfactual'][2]
lo=[c['truth_delta_y_evaluator_m']*1000 for c in e['low']['candidates']];hi=[c['truth_delta_y_evaluator_m']*1000 for c in e['high']['candidates']]
panel(465,'Matched pull: true displacement (mm)',[lo,hi],['resistance 0.01 N','resistance 0.04 N'],-15,15)
s.append('<text x="30" y="395" font-size="10">Simulator curves use evaluator-only truth; candidate scores are image-derived.</text></svg>')
with (out/'decision_curves.svg').open('x') as f:f.write('\n'.join(s))
