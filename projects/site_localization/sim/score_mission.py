"""Post-run physical outcome judge. No estimator/controller inputs produced."""
import json
from pathlib import Path
import sys
import numpy as np


def score(out):
    cfg=json.loads((out/'cell.json').read_text())
    rows=[json.loads(s) for s in (out/'judge.jsonl').read_text().splitlines()]
    truth=[r for r in rows if r['type']=='truth'];contacts=[r for r in rows if r['type']=='contacts']
    if len(truth)<2:raise ValueError('Missing physical trajectory; cannot judge mission')
    t=np.array([r['time'] for r in truth]);q=np.array([r['q'] for r in truth])
    yaw=np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
    poses=np.c_[[r['x'] for r in truth],[r['y'] for r in truth],yaw]
    goals=np.array(cfg['goals']);completed=[];dwell=None;goal=0
    for i in range(1,len(t)):
        if goal==len(goals):break
        dt=t[i]-t[i-1]
        if dt<=0:continue
        delta=poses[i]-poses[i-1];delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
        err=poses[i]-goals[goal];err[2]=np.arctan2(np.sin(err[2]),np.cos(err[2]))
        acceptable=(np.linalg.norm(err[:2])<=cfg['judge_position_m'] and abs(err[2])<=cfg['judge_yaw_rad']
                    and np.linalg.norm(delta[:2])/dt<.025 and abs(delta[2])/dt<.05 and dt<.15)
        if acceptable:
            if dwell is None:dwell=t[i]
            if t[i]-dwell>=cfg['dwell_s']:
                completed.append(dict(goal=goal,time=float(t[i]),position_error_m=float(np.linalg.norm(err[:2])),
                                      yaw_error_deg=float(np.rad2deg(abs(err[2])))))
                goal+=1;dwell=None
        else:dwell=None
    contact_events=[r for r in contacts if r['count']]
    wheel=[r for r in rows if r['type']=='wheel_contacts' and r['count']]
    contact_live=bool(wheel) and wheel[-1]['time']>=t[-1]-.25
    # Independently check conservative circular footprint against declared world geometry.
    clearance=[]
    for ox,oy,sx,sy in cfg['obstacles']:
        dx=np.maximum(abs(poses[:,0]-ox)-sx/2,0);dy=np.maximum(abs(poses[:,1]-oy)-sy/2,0)
        clearance.append(np.hypot(dx,dy)-.23)
    minimum_clearance=float(np.min(clearance))
    success=(goal==len(goals) and not contact_events and contact_live and minimum_clearance>0
             and completed[-1]['time']<=cfg['max_sim_s'])
    summary=dict(split='development',scenario=cfg['scenario'],mission_success=bool(success),
                 physical_stops_completed=len(completed),required_stops=len(goals),stops=completed,
                 truth_samples=len(truth),contact_messages=len(contacts),body_contact_messages=len(contact_events),
                 wheel_contact_heartbeat_messages=len(wheel),contact_system_live=bool(contact_live),
                 minimum_conservative_clearance_m=minimum_clearance,observed_sim_span_s=float(t[-1]-t[0]),
                 final_truth_pose=poses[-1].tolist(),
                 limitations=['one noiseless development world','analytic commissioned prior',
                              'simple waypoint controller; no path replanning','no map repair yet'])
    (out/'outcome.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':score(Path(sys.argv[1]))
