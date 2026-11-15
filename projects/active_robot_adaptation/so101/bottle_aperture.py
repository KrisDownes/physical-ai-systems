"""Evaluator-only preclosure geometric-clearance diagnostic."""
import gzip,json,sys
from pathlib import Path
import mujoco
from bottle_task import BottleEnvironment
from run_store import create_run
p=Path(sys.argv[1]);files=sorted((p/'episodes').glob('*.json'))
if files:rows=[json.loads(x.read_text()) for x in files]
else:
 with gzip.open(p/'episodes_full.json.gz','rt') as f:rows=json.load(f)
out=create_run('bottle_aperture',{'source':str(p),'note':'truth for offline diagnostics only'});scores=[]
for r in rows:
 s=r['result'].get('evaluation',{}).get('snapshots',{}).get('before_close')
 if not s or s['tilt_deg']>.1:continue
 e=BottleEnvironment(r['condition']['fill'],r['condition']['grip'],render=False);e.reset();e.data.qpos[:6]=s['joints_rad'];e.data.qpos[6:9]=s['object_xyz'];e.data.qpos[9:13]=[1,0,0,0];mujoco.mj_forward(e.model,e.data)
 clearance={}
 for body in ('gripper','moving_jaw_so101_v1'):
  ds=[]
  for i in range(e.model.ngeom):
   b=mujoco.mj_id2name(e.model,mujoco.mjtObj.mjOBJ_BODY,int(e.model.geom_bodyid[i]))
   if b==body and e.model.geom_contype[i]:ds.append(mujoco.mj_geomDistance(e.model,e.data,e.geom,i,.1,None))
  clearance[body]=min(ds)
 scores.append({'case':r['case']['id'],'condition':r['condition'],'clearance_m':clearance,'preclose_contacts':s['contacts']});e.close()
with (out/'clearances.json').open('x') as f:json.dump(scores,f,indent=2)
print('saved',out,'cases',len(scores),'minimum clearance mm',min(min(s['clearance_m'].values()) for s in scores)*1000)
