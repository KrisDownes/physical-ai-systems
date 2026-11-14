"""Make a dependency-free, per-episode evaluator diagnostic plot."""
import json,sys,html
from pathlib import Path
p=Path(sys.argv[1]); rows=json.loads((p/'transfer_metrics.json').read_text()); width,height=700,420
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="420" viewBox="0 0 700 420">','<rect width="700" height="420" fill="white"/>','<text x="60" y="25" font-size="17">Grasp prediction error and settled placement</text>', '<path d="M60 350 H650 M60 350 V50" fill="none" stroke="black"/>']
for y in [0,10,20,30,35]:
 py=350-y*8;color="#b00" if y==35 else "#ddd";parts.append(f'<path d="M60 {py} H650" stroke="{color}" stroke-dasharray="3 3"/><text x="27" y="{py+4}" font-size="11">{y}</text>')
for x in [0,2,4,6,8]:
 px=60+x*70;parts.append(f'<text x="{px-3}" y="367" font-size="11">{x}</text>')
for row in rows:
 if row.get('grasp_relative_prediction_error_mm') is None or row.get('final_placement_error_mm') is None:continue
 x=60+row['grasp_relative_prediction_error_mm']*70;y=350-row['final_placement_error_mm']*8;color='#1757b4' if row['references']=='visible' else '#d07600'
 title=html.escape(row['condition']+' '+row['case']);parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" fill-opacity=".75"><title>{title}</title></circle>')
parts += ['<text x="170" y="400" font-size="13">Grasp-relative prediction error at closure (mm)</text>','<text transform="translate(15 320) rotate(-90)" font-size="13">Final placement error (mm)</text>','<text x="405" y="48" fill="#1757b4" font-size="12">● table visible</text>','<text x="510" y="48" fill="#d07600" font-size="12">● table unavailable</text>','<text x="505" y="69" fill="#b00" font-size="11">35 mm success threshold</text>','</svg>']
with (p/'transfer_scatter.svg').open('x') as f:f.write('\n'.join(parts))
