"""Bottle grasp-geometry development sweep; not a research comparison."""
import json
from bottle_task import BottleEnvironment
from environment import Q_HOME
from nominal import run
from run_store import create_run

config={'fill':'full','grip':'dry','object_xy':[.217,.03],'pan':-.50,'seed':2111,'offsets':[[x,y] for x in [-.020,-.014,-.008] for y in [-.035,-.030]]}
out=create_run('bottle_geometry_development',config);rows=[]
for offset in config['offsets']:
 e=BottleEnvironment('full','dry',action_budget=200,movement_budget=8);q=Q_HOME.copy();q[0]=config['pan']
 r=run(e,config['object_xy'],q,method='calibration',seed=config['seed'],grasp_offset=offset);r.pop('frames',None)
 row={'offset':offset,'result':r,'telemetry_evaluator':e.telemetry,'trace':e.trace};rows.append(row);e.close()
 print(offset,r['success'],r.get('failure'),flush=True)
with (out/'results.json').open('x') as f:json.dump(rows,f,indent=2)
print('saved',out)
