"""Analyze the previously recorded fixed/online counterexample without retuning."""
import gzip
import json
import numpy as np
from pathlib import Path
from run_store import create_run

SOURCE=Path('results/so101_runs/20261114T015619.662436Z_49ac12f199cf_fresh_camera_followup_1aff5aad/episodes_full.json.gz')
data=json.load(gzip.open(SOURCE,'rt'))
rows={m:next(r for r in data if r['case']['id']=='heldout-a' and r['condition']=='visible' and r['method']==m)
      for m in ('nominal','online')}
out=create_run('online_discrepancy',{'case':'heldout-a','source':str(SOURCE),'seeds':[811]})
summary={'source':str(SOURCE),'case':rows['nominal']['case'],'methods':{}}
for method,r in rows.items():
    t=r['trace']; last=max(i for i,x in enumerate(t) if x.get('stage')=='reach')
    closure=t[last+1:last+16]
    summary['methods'][method]={'success':r['result']['success'],'failure':r['result']['failure'],
      'target_xy':r['result']['localization']['grasp_target_xy'],
      'last_reach_joint_rad':t[last]['joint_rad'],'last_reach_site_xyz':t[last]['site_xyz_evaluator'],
      'last_reach_object_xyz':t[last]['object_xyz_evaluator'],
      'object_xyz_after_closure':r['result']['evaluation']['snapshots']['after_close']['object_xyz'],
      'jaw_contact':r['result']['evaluation']['grasp_contact'],
      'jaw_after_closure_rad':closure[-1]['joint_rad'][5],
      'saturated_actions':r['saturated_actions'],'total_actions':r['actions'],
      'updates_before_grasp':sum(bool(x.get('adaptation',{}).get('accepted')) for x in t[:last+1])}
    if method=='online': summary['methods'][method]['first_update']=t[0]['adaptation']
a,b=(rows[x]['trace'] for x in ('nominal','online'))
summary['paired_steps']=[{'step':i,'stage':x.get('stage'),
   'command_difference_rad':float(np.linalg.norm(np.asarray(x['sent_delta_rad'])-y['sent_delta_rad'])),
   'tip_difference_m':float(np.linalg.norm(np.asarray(x['site_xyz_evaluator'])-y['site_xyz_evaluator'])),
   'object_difference_m':float(np.linalg.norm(np.asarray(x['object_xyz_evaluator'])-y['object_xyz_evaluator']))}
   for i,(x,y) in enumerate(zip(a,b)) if i<27]
(out/'discrepancy.json').write_text(json.dumps(summary,indent=2)+'\n')
print('saved',out)
