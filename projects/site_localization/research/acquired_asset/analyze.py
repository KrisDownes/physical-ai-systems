"""Independent acquisition/inspection analysis; truth is never a policy input."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/site_acquired_asset_plot')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def angle(a):return np.arctan2(np.sin(a),np.cos(a))
def raster(path):
    with path.open() as f:
        w,h,r,x,y=np.fromstring(f.readline(),sep=' ');a=np.loadtxt(f,dtype=int).reshape(int(h),int(w))
    return a,[x,x+w*r,y,y+h*r],r
def reference(folder):
    data=[r for r in rows(folder/'judge.jsonl') if r['type']=='truth']
    q=np.array([r['q'] for r in data]);yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
    t=np.array([r['time'] for r in data]);xy=np.array([[r['x'],r['y']] for r in data])
    return t,xy,yaw
def write_csv(path,data):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def analyze(root,out):
    out.mkdir(parents=True,exist_ok=False);missions=[];acquisitions=[];details=[]
    project=Path(__file__).resolve().parents[2];freeze=load(root/'freeze.json')
    for folder in sorted(root.glob('acquisition__*')):
        if not folder.is_dir() or not(folder/'run.json').exists():continue
        run=load(folder/'run.json');judge=rows(folder/'judge.jsonl');t,xy,yaw=reference(folder)
        result=load(folder/'outcome.json') if (folder/'outcome.json').exists() else {}
        done=load(folder/'controller_done.json') if (folder/'controller_done.json').exists() else {}
        exported=(folder/'map_in_site.txt').exists()
        acquisitions.append(dict(episode=folder.name,controller_completed=run['stop_reason']=='controller_completed',
            map_transferred=exported,route_physical_stops=result.get('physical_stops_completed'),
            legacy_route_contract_success=result.get('mission_success'),dock_success=result.get('dock_return_success'),
            physical_path_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
            simulation_end_s=float(t[-1]),controller_done_s=done.get('time'),wall_s=run['wall_s'],
            contact_events=sum(r['type']=='contacts' and r['count']>0 for r in judge),
            scans=load(folder/'mapper/bridge_summary.json')['scan_messages'] if (folder/'mapper/bridge_summary.json').exists() else None,
            stop_reason=run['stop_reason'],all_owned_processes_reaped=run['all_owned_processes_reaped']))
    for case in freeze['inspection_order']:
        for treatment in ['supplied','acquired']:
            folder=root/(case+'__'+treatment)
            if not(folder/'outcome.json').exists():continue
            result=load(folder/'outcome.json');t,xy,yaw=reference(folder)
            history=np.array(rows(folder/'method_pose_history.jsonl'))
            supported=(history[:,0]>=t[0])&(history[:,0]<=t[-1]);h=history[supported]
            gt=np.c_[np.interp(h[:,0],t,xy[:,0]),np.interp(h[:,0],t,xy[:,1]),np.interp(h[:,0],t,yaw)]
            position=np.linalg.norm(h[:,1:3]-gt[:,:2],axis=1);heading=abs(angle(h[:,3]-gt[:,2]))
            # Correlated time samples are descriptive errors, not independent deployments.
            events=rows(folder/'method.jsonl');sensor=rows(folder/'private/sensor_truth.jsonl')
            decoded=[e for e in events if e['event']=='observation' and 7 in e['vision'].get('observed_ids',[])]
            uncertain=[e for e in events if e['event']=='observation' and e['vision']['identity_decision'].startswith('target_pose_uncertain')]
            observed=[]
            for e in events:
                if e['event']!='observation' or not e.get('observed_asset_map'):continue
                truth=next(s for s in sensor if s['id']==e['id']);asset=next(a for a in truth['assets'] if a['actual_id']==7)
                target=np.array(asset['asset_pose'][:2]);p=np.array(e['observed_asset_map']['position'])
                rp=np.array(e['associated_pose']);actual=np.array(truth['robot_pose'])
                v=e['vision']['target'];x,y,z=v['gauge_camera'];a=actual[2]
                oracle=actual[:2]+np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@np.array([z,-x])
                observed.append(dict(capture=e['id'],stamp=e['sensor_time'],associated_pose=e['associated_pose'],
                    actual_robot_pose=truth['robot_pose'],camera_gauge=v['gauge_camera'],
                    observed_asset=e['observed_asset_map'],actual_asset=asset['asset_pose'],
                    map_target_error_m=float(np.linalg.norm(p-target)),
                    pixel_relative_error_oracle_robot_m=float(np.linalg.norm(oracle-target)),
                    localization_position_error_m=float(np.linalg.norm(rp[:2]-actual[:2])),
                    localization_heading_error_deg=float(np.degrees(angle(rp[2]-actual[2]))),
                    processing_age_s=e['decision_time']-e['sensor_time'],reprojection_rms_px=v['rms_px']))
            plans=[e for e in events if e['event']=='plan' and e['purpose'] in ['approach','recovery']]
            details.append(dict(mission=folder.name,observations=observed,approach_plans=plans,
                physical_observations=result['observations'],search_positions=result['search_positions']))
            missions.append(dict(mission=folder.name,case=case,treatment=treatment,
                completed=result['correct_physical_task_completion'],claim=result['success_claim'],
                false_claim=result['false_completion_claim'],wrong_asset_acceptances=result['wrong_asset_acceptances'],
                contact_events=result['contact_events'],clearance_m=result['minimum_clearance_m'],
                navigation_failure=result['navigation_failure'],infrastructure_failure=result['infrastructure_failure'],
                simulation_s=result['simulation_time_s'],path_m=result['physical_path_m'],captures=result['captures'],
                first_detection_s=result['first_target_detection_sim_s'],first_detection_path_m=result['distance_to_first_target_detection_m'],
                first_decoded_target_id_s=decoded[0]['sensor_time'] if decoded else None,
                rejected_target_pose_observations=len(uncertain),
                recoveries=result['recovery_actions'],reading_abstentions=result['inspection_abstentions'],
                localization_position_rms_m=float(np.sqrt(np.mean(position**2))),
                localization_position_max_m=float(position.max()),
                localization_heading_rms_deg=float(np.degrees(np.sqrt(np.mean(heading**2)))),
                first_target_error_m=observed[0]['map_target_error_m'] if observed else None,
                last_target_error_m=observed[-1]['map_target_error_m'] if observed else None,
                approach_feasible=all(p['path'] is not None for p in plans) if plans else None,
                reason=result['termination_reason']))
    if acquisitions:write_csv(out/'acquisitions.csv',acquisitions)
    if missions:write_csv(out/'missions.csv',missions)
    save(out/'diagnostics.json',details)
    matched={}
    for case in freeze['inspection_order']:
        a=root/(case+'__supplied');b=root/(case+'__acquired')
        if not (b/'outcome.json').exists():continue
        names=['controller_config.json','public/policy.json','public/vision.json','private/scene.json','private/world.sdf','private/evaluation.json']
        matched[case]={n:sha(a/n)==sha(b/n) for n in names}
    before=load(root.parent/'preservation_before.json')
    audit=dict(freeze_sha256=sha(root/'freeze.json'),protected_files=len(before),
        protected_mismatches=[p for p,h in before.items() if sha(project/p)!=h],
        frozen_mismatches=[p for p,h in freeze['files'].items() if sha(Path(p))!=h],
        matched_inputs_except_map=matched)
    save(out/'audit.json',audit)
    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for i,condition in enumerate(['unchanged','moved']):
        for j,treatment in enumerate(['supplied','acquired']):
            folder=root/(condition+'__'+treatment);ax=axes[i,j]
            if not (folder/'map.txt').exists():continue
            a,extent,res=raster(folder/'map.txt');image=np.where(a==1,0,np.where(a==-1,1,.6))
            ax.imshow(image,origin='lower',extent=extent,cmap='gray',vmin=0,vmax=1)
            t,xy,y=reference(folder);ax.plot(xy[:,0],xy[:,1],color='#1177bb',label='physical trajectory')
            h=np.array(rows(folder/'method_pose_history.jsonl'));ax.plot(h[:,1],h[:,2],color='#db6644',alpha=.7,lw=.8,label='AMCL')
            ax.set(xlim=(-3.4,3.4),ylim=(-3.4,3.4),title=condition+' / '+treatment,xlabel='site x (m)',ylabel='site y (m)')
            ax.legend(fontsize=8);ax.set_aspect('equal')
    fig.suptitle('Acquired-map comparison: raster plus independent physical and estimated paths')
    fig.tight_layout();fig.savefig(out/'maps_and_paths.png',dpi=130);plt.close(fig)
    print(json.dumps(dict(acquisitions=len(acquisitions),missions=len(missions),audit=audit),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.runs,a.out)
