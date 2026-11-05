"""Independent physical outcomes plus failed-target/effort accounting."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main(out):
    spec=importlib.util.spec_from_file_location('original_judge',ROOT/'sim/score_mission.py')
    judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)
    with contextlib.redirect_stdout(io.StringIO()):result=judge.score(out)
    cfg=json.loads((out/'cell.json').read_text());rows=[json.loads(x) for x in (out/'judge.jsonl').read_text().splitlines()]
    truth=[r for r in rows if r['type']=='truth'];times=np.array([r['time'] for r in truth])
    q=np.array([r['q'] for r in truth]);yaw=np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
    xy=np.array([[r['x'],r['y']] for r in truth]);claims=[r['event'] for r in rows if r['type']=='event' and r['event']['type']=='claimed_stop']
    errors=[]
    for claim in claims:
        t=claim['time'];target=np.array(cfg['goals'][claim['goal']])
        hi=int(np.searchsorted(times,t,side='right'));lo=max(0,hi-1)
        supported=(times[0]<=t<=times[-1] and (hi==len(times) or times[hi]-times[lo]<=.15))
        if not supported:
            errors.append(dict(goal=claim['goal'],claim_time_s=t,reference_supported=False,
                               physical_position_error_m=None,physical_heading_error_deg=None))
            continue
        p=np.array([np.interp(t,times,xy[:,0]),np.interp(t,times,xy[:,1])])
        angle=np.interp(t,times,np.unwrap(yaw));d=np.arctan2(np.sin(angle-target[2]),np.cos(angle-target[2]))
        errors.append(dict(goal=claim['goal'],reference_supported=True,physical_position_error_m=float(np.linalg.norm(p-target[:2])),
                           physical_heading_error_deg=float(np.rad2deg(abs(d))),claim_time_s=t))
    target_metrics=[]
    for i,target in enumerate(cfg['goals']):
        distance=np.linalg.norm(xy-np.array(target[:2]),axis=1);angle=abs(np.arctan2(np.sin(yaw-target[2]),np.cos(yaw-target[2])))
        valid=(distance<=cfg['judge_position_m'])&(angle<=cfg['judge_yaw_rad']);best_dwell=0.;start=None
        for j in range(1,len(times)):
            dt=times[j]-times[j-1];motion=np.linalg.norm(xy[j]-xy[j-1])/max(dt,1e-12)
            rotation=abs(np.arctan2(np.sin(yaw[j]-yaw[j-1]),np.cos(yaw[j]-yaw[j-1])))/max(dt,1e-12)
            if valid[j] and 0<dt<.15 and motion<.025 and rotation<.05:
                if start is None:start=times[j]
                best_dwell=max(best_dwell,times[j]-start)
            else:start=None
        nearest=int(np.argmin(distance))
        target_metrics.append(dict(goal=i,region=cfg.get('inspection_zones',['unchanged','unchanged','changed_sensitive','changed_sensitive'])[i],
            min_position_error_m=float(distance[nearest]),heading_at_nearest_deg=float(np.rad2deg(angle[nearest])),
            longest_valid_stationary_dwell_s=float(best_dwell),dwell_deficit_s=float(max(0,cfg['dwell_s']-best_dwell))))
    result.update(condition=cfg['condition'],remedy=cfg['remedy'],localizer=cfg['localizer'],mapping=cfg['mapping'],
        map_from_mission=cfg['map_from_mission'],claimed_stop_errors=errors,target_metrics=target_metrics,
        operator_events=cfg['operator_events'],human_operator_time_measured=False,reset_count=cfg['reset_count'],
        method_truth_inputs=False,initialization_assistance=cfg['initialization'],
        diagnostic_dwell_scope='Whole trajectory per target; ordered completion remains stops/mission_success',
        compute_instrumentation='All modes include diagnostic native AMCL, even when another localizer controls',
        limitations=['one development layout','no injected sensor noise','known-dock initialization',
                     'waypoint controller; changes stay off required routes','no physical transfer claim'])
    if cfg.get('dock_completion'):
        dock=cfg['dock_completion'];target=np.array(dock['pose']);start=None;completion=None
        after=result['stops'][-1]['time'] if result['physical_stops_completed']==len(cfg['goals']) else float('inf')
        for j in range(1,len(times)):
            dt=times[j]-times[j-1]
            angle=abs(np.arctan2(np.sin(yaw[j]-target[2]),np.cos(yaw[j]-target[2])))
            rotation=abs(np.arctan2(np.sin(yaw[j]-yaw[j-1]),np.cos(yaw[j]-yaw[j-1])))/max(dt,1e-12)
            valid=(times[j]>after and 0<dt<.15 and np.linalg.norm(xy[j]-target[:2])<=dock['position_tolerance_m']
                   and angle<=dock['yaw_tolerance_rad'] and np.linalg.norm(xy[j]-xy[j-1])/max(dt,1e-12)<.025 and rotation<.05)
            if valid:
                if start is None:start=times[j]
                if times[j]-start>=dock['stationary_dwell_s']:completion=float(times[j]);break
            else:start=None
        result['dock_completion_time_s']=completion
        result['dock_return_success']=completion is not None and completion<=cfg['max_sim_s']
        result['inspection_success']=result['mission_success']
        result['mission_success']=result['mission_success'] and result['dock_return_success']
    result['max_sim_s']=cfg['max_sim_s']
    result['claimed_stops_after_deadline']=sum(e['claim_time_s']>cfg['max_sim_s'] for e in errors)
    (out/'outcome.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
