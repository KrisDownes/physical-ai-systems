"""Compact held-out one-step error plot; per-seed values, not independent task trials."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text());cur=s['prediction']['current']['test_rmse_mm_per_coordinate'];hist=s['prediction']['history']['test_rmse_mm_per_coordinate']
out=['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="360">','<rect width="650" height="360" fill="white"/>','<text x="45" y="26" font-size="17">Held-out prediction error across training seeds</text>','<path d="M55 285 H620 M55 60 V285" fill="none" stroke="black"/>']
for n in range(6):out.append(f'<text x="32" y="{290-n*40}" font-size="12">{n}</text><path d="M55 {285-n*40} H620" stroke="#ddd"/>')
for i in range(3):
 for j,(values,color) in enumerate(((cur,'#2360ac'),(hist,'#ce7a18'))):
  v=values[i];x=110+i*155+j*40;y=285-v*40;out.append(f'<rect x="{x}" y="{y:.2f}" width="32" height="{v*40:.2f}" fill="{color}"/><text x="{x}" y="{y-5:.2f}" font-size="11">{v:.2f}</text>')
 out.append(f'<text x="{110+i*155}" y="305" font-size="12">seed {i+1}</text>')
out+=['<text x="75" y="49" fill="#2360ac" font-size="12">■ current only</text><text x="200" y="49" fill="#ce7a18" font-size="12">■ history</text>','<text x="55" y="340" font-size="12">One-step RMSE, mm per coordinate; 40 transitions from four episodes.</text>','</svg>']
with (p/'prediction.svg').open('x') as f:f.write('\n'.join(out))
