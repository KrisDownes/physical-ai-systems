"""Two open changed-site cases, shared candidates and frozen conventional rules."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import time
import numpy as np
from frames import compose, inverse
from transfer_surface_diagnostic import boundary_map
from transfer_layouts import make
from review_matched_pair import sha, load, write, now, select_saved

ROOT=Path(__file__).resolve().parents[1]


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    names=['register_maps.py','transfer_surface_diagnostic.py','controller.py','run.py',
           'frames.py','score.py','corrected_trial.py','transfer_layouts.py','review_matched_pair.py']
    files=[ROOT/'research'/n for n in names]+[ROOT/'sim/score_mission.py',ROOT/'sim/judge_node.py',ROOT/'sim/cell.sdf',ROOT/'vendor/offline_amcl',Path(__file__)]
    old=load(ROOT/'results/review_matched_pair_20261023/candidate_verification.json')['source_hashes']
    for p in files:
        if str(p) in old:assert sha(p)==old[str(p)]
    config=dict(registered_utc=now(),source_hashes={str(p):sha(p) for p in files},
        cap_m=.3,trim_fraction=.7,max_correspondence_m=.15,huber_m=.08,
        starts=27,translation_radius_m=.25,yaw_radius_rad=.1,iterations=100,
        max_points=12000,inlier_distance_m=.12,translation_tolerance_m=1e-5,yaw_tolerance_rad=1e-5,
        materiality=dict(max_route_or_dock_position_disagreement_m=.005,yaw_disagreement_rad=.001,
            meaning='Predeclared execution triage only, not acceptance: 5mm is one tenth of a grid cell; 0.001rad is 0.057 degrees.'),
        physical_contract=dict(position_m=.15,heading_deg=10,dwell_s=2,ordered_visits=6,dock=True,positive_clearance=True,body_contacts=0,max_sim_s=500),
        arrival_tolerance_m=.03,control_dwell_margin_s=.2,wall_limit_s=620,
        execution='Sequential, only materially different selected transforms; no duplicate agreement missions. No automatic retries.',
        competing_predictions=['All-source support preserves useful transverse geometry.','Changed surfaces bias all-source scoring toward an incorrect transform.'],
        scope='Two already-open maintenance development acquisitions; no reserved access.')
    write(out/'freeze.json',config)
    rows=[];jobs=[]
    for condition in ['additions','rearrangement']:
        acq=ROOT/f'results/transfer_20261022_A/maintenance_bays__{condition}__full_acquisition'
        cfg=load(acq/'cell.json');dest=out/condition;dest.mkdir()
        provenance=dict(acquisition=str(acq),seed=cfg['seed'],map_sha256=sha(acq/'mapper/map.txt'),historical_map_sha256=sha(acq/'map.txt'),
            source=boundary_map(acq/'mapper/map.txt',dest/'source_surface.txt'),target=boundary_map(acq/'map.txt',dest/'target_surface.txt'))
        write(dest/'inputs.json',provenance)
        cmd=['/usr/bin/python3',str(ROOT/'research/register_maps.py'),'--source',str(dest/'source_surface.txt'),'--target',str(dest/'target_surface.txt'),
            '--output',str(dest/'alignment.json'),'--prior','-6','-.3','0','--method','robust','--selection-objective','trimmed',
            '--max-correspondence','.15','--trim-fraction','.7','--huber-delta','.08','--inlier-distance','.12',
            '--translation-radius','.25','--yaw-radius','.1','--max-points','12000','--score-cap','.3','--iterations','100',
            '--translation-tolerance','.00001','--yaw-tolerance','.00001']
        t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        (dest/'registration.stdout').write_text(r.stdout);(dest/'registration.stderr').write_text(r.stderr)
        write(dest/'generation.json',dict(command=cmd,returncode=r.returncode,wall_s=time.monotonic()-t))
        assert r.returncode==0
        a=load(dest/'alignment.json');assert len(a['candidates'])==27
        selections={}
        for objective,key in [('trimmed','trimmed_objective_m2'),('capped','objective_m2')]:
            i=min(range(27),key=lambda i:a['candidates'][i][key]);selections[objective]=select_saved(a,objective,i)
        fixture=make(dest/'fixture','maintenance_bays',condition,'acquired_map')
        poses=np.array([fixture['start']]+[x['pose'] for x in fixture['route']])
        t0=selections['trimmed']['selected']['pose'];t1=selections['capped']['selected']['pose']
        map_poses=compose(inverse(t0),poses)
        delta=compose(t1,map_poses)-poses
        maximum=float(np.linalg.norm(delta[:,:2],axis=1).max());yaw=float(abs(np.arctan2(np.sin(t0[2]-t1[2]),np.cos(t0[2]-t1[2]))))
        material=maximum>=.005 or yaw>=.001
        for objective,s in selections.items():
            transform=inverse(s['selected']['pose']).tolist();s.update(target_from_source=s['selected']['pose'],map_from_mission=transform,
                map_sha256=provenance['map_sha256'],shared_alignment_sha256=sha(dest/'alignment.json'),
                map_initial_pose=compose(transform,fixture['start']).tolist(),map_route_poses=compose(transform,np.array([x['pose'] for x in fixture['route']])).tolist())
            write(dest/(objective+'_selection.json'),s)
            if material:
                path=out/f'maintenance_bays__{condition}__{objective}__deployment'
                command=['/usr/bin/python3',str(ROOT/'research/corrected_trial.py'),'--fixture-layout','maintenance_bays','--fixture-condition',condition,
                    '--split','open_changed_development','--frame-policy','surface_fit_diagnostic','--out',str(path),'--seed',str(cfg['seed']),
                    '--domain','199','--wall-limit','620','--remedy','acquired_map','--map',str(acq/'mapper/map.txt'),
                    '--map-from-mission',*map(str,transform)]
                jobs.append(dict(condition=condition,objective=objective,output=str(path),command=command,selection=str(dest/(objective+'_selection.json'))))
        rows.append(dict(condition=condition,seed=cfg['seed'],selections={k:dict(index=s['selected_index'],pose=s['selected']['pose'],trimmed_m2=s['selected']['trimmed_objective_m2'],capped_m2=s['selected']['objective_m2']) for k,s in selections.items()},
            max_route_or_dock_disagreement_m=maximum,yaw_disagreement_rad=yaw,materially_different=material,
            shared_alignment_sha256=sha(dest/'alignment.json'),map_sha256=provenance['map_sha256']))
    write(out/'selections_before_outcomes.json',dict(recorded_utc=now(),cases=rows,jobs=jobs,new_physical_outcomes_inspected=False))
    print(json.dumps(rows,indent=2))


def execute(out):
    import os,signal
    freeze=load(out/'freeze.json')
    for p,h in freeze['source_hashes'].items():assert sha(p)==h,p
    plan=load(out/'selections_before_outcomes.json');assert not (out/'attempts.json').exists()
    attempts=[]
    for j in plan['jobs']:
        p=Path(j['output']);assert not p.exists()
        e=dict(condition=j['condition'],objective=j['objective'],command=j['command'],output=str(p),started_utc=now(),state='launching',reset_requested=1)
        attempts.append(e);write(out/'attempts.json',attempts);start=time.monotonic()
        with (out/f"{j['condition']}_{j['objective']}.log").open('w') as log:
            proc=subprocess.Popen(j['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);e.update(pid=proc.pid,state='running');write(out/'attempts.json',attempts)
            try:e['returncode']=proc.wait(timeout=750)
            except subprocess.TimeoutExpired:
                e['infrastructure_issue']='parent timeout; SIGINT cleanup';os.killpg(proc.pid,signal.SIGINT);e['returncode']=proc.wait(timeout=60)
        (p/'calibration_selection.json').write_bytes(Path(j['selection']).read_bytes())
        r=load(p/'run.json');e.update(state='finished',finished_utc=now(),wall_s=time.monotonic()-start,reset_count=r['reset_count'],physics_started=r['physics_started'],all_processes_reaped=r['all_owned_processes_reaped'],score_returncode=r['score_returncode'],stop_reason=r['stop_reason'])
        write(out/'attempts.json',attempts);print(json.dumps(e),flush=True)
        assert r['physics_started'] and r['all_owned_processes_reaped'] and r['score_returncode']==0,'Infrastructure failure retained; stop for diagnosis'
    write(out/'collection_complete.json',dict(finished_utc=now(),attempts=len(attempts),new_missions=len(attempts),reserved_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare)(a.out.resolve())
