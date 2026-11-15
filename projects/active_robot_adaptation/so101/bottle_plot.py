"""Small dependency-free chart from paired evaluator records."""
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=json.loads((p/'paired.json').read_text());out=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360">','<rect width="760" height="360" fill="white"/>','<text x="50" y="27" font-size="18">Settled bottle placement, three paired starts</text>','<path d="M55 285 H735 M55 65 V285" stroke="black" fill="none"/>']
for i,fill in enumerate(('empty','half','full')):
 for j,grip in enumerate(('dry','wet')):
  g=[r for r in rows if r['condition']=={'fill':fill,'grip':grip}];x=75+(i*2+j)*108
  for k,method in enumerate(('baseline','conservative')):
   n=sum(r[method]['success'] for r in g);h=n*65;color='#2360ac' if k==0 else '#ce7a18'
   out.append(f'<rect x="{x+30*k}" y="{285-h}" width="24" height="{h}" fill="{color}"/><text x="{x+30*k+7}" y="{278-h}" font-size="12">{n}</text>')
  out.append(f'<text x="{x-2}" y="305" font-size="11">{fill} {grip}</text>')
for n in range(4):out.append(f'<text x="35" y="{290-n*65}" font-size="12">{n}</text>')
out+=['<text x="400" y="48" font-size="12" fill="#2360ac">■ baseline</text><text x="505" y="48" font-size="12" fill="#ce7a18">■ max closure + slow</text>','<text x="55" y="338" font-size="12">Counts are paired, not a general ranking; 18 episodes per method.</text>','</svg>']
with (p/'success.svg').open('x') as f:f.write('\n'.join(out))
