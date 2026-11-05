"""Post-run truth diagnostics only. Never supplies inputs to any robot process."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from prepare import floor_grid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import csv_pose, error_summary, truth_pose
sys.path.insert(0,str(ROOT/'research/moved_asset'))
from render import blocked_segments
sys.path.insert(0,str(ROOT/'research/continuous_asset'))
from buffered_planning import BufferedPlanner


def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def map_mask(points,path,frame,center_header=False):
    with path.open() as f:
        w,h,res,ox,oy=np.fromstring(f.readline(),sep=' ');grid=np.loadtxt(f,dtype=int)
    c,s=math.cos(frame[2]),math.sin(frame[2]);v=points-np.array(frame[:2])
    source=np.c_[c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1]]
    position=(source-[ox,oy])/res
    index=(np.rint(position) if center_header else np.floor(position)).astype(int)
    valid=(index[:,0]>=0)&(index[:,1]>=0)&(index[:,0]<w)&(index[:,1]<h)
    free=np.zeros(len(points),bool);known=np.zeros(len(points),bool)
    states=grid[index[valid,1],index[valid,0]]
    free[valid]=states==-1;known[valid]=states!=0
    return free,known


def visual_mask(points,pose,scene):
    camera=np.array([*pose[:2],scene['camera']['height_m']])
    v=points-np.array(pose[:2]);c,s=math.cos(pose[2]),math.sin(pose[2])
    forward=c*v[:,0]+s*v[:,1];right=s*v[:,0]-c*v[:,1]
    half=math.radians(scene['camera']['horizontal_fov_deg']/2)
    possible=(forward>.05)&(abs(right)<forward*math.tan(half))
    indices=np.flatnonzero(possible)
    points3=np.c_[points[indices],np.full(len(indices),scene['asset_height_m'])]
    possible[indices]&=~blocked_segments(camera,points3,scene['obstacles'])
    return possible


def analyze(root,partial=False):
    outdir=root/'analysis';outdir.mkdir(exist_ok=True)
    scenarios=load(root/'execution_freeze.json')['cases']
    preflight={r['case']:r for r in load(root/'evaluator_preflight/feasibility.json')['cases']}
    ledger=[];diagnostics={};plotdata={}
    for case in scenarios:
        out=root/('mission_'+case)
        if not (out/'outcome.json').exists():
            if partial:continue
            raise RuntimeError('Incomplete attempt: '+case)
        result=load(out/'outcome.json');run=load(out/'run.json');scene=load(out/'private/scene.json')
        truth=truth_pose(out);acq=result.get('acquisition',{})
        axis,points,free,connected,clearance=floor_grid(scene['obstacles']);mask=connected.ravel()
        sensors=rows(out/'inspection/private/sensor_truth.jsonl');events=rows(out/'inspection/method.jsonl')
        exploration=rows(out/'acquisition/exploration.jsonl')
        handoff=load(out/'handoff.json') if (out/'handoff.json').exists() else None
        frame=handoff['site_from_slam'] if handoff else [-2,-2,0]
        actual_map=out/'acquisition/mapper/map.txt'
        if actual_map.exists():mapped,known=map_mask(points,actual_map,frame)
        else:mapped=known=np.zeros(len(points),bool)
        seen=np.zeros(len(points),bool)
        for sensor in sensors:seen|=visual_mask(points,sensor['robot_pose'],scene)
        coverage={}
        for name,region in dict(all=mask,eastern_room=mask&(points[:,0]>.59),
                northwest_bay=mask&(points[:,0]<-1.5)&(points[:,1]>.59)).items():
            coverage[name]=dict(reference_floor_area_m2=float(region.sum()*.0025),
                represented_free_area_m2=float((mapped&region).sum()*.0025),
                mapped_fraction=float((mapped&region).sum()/region.sum()),
                captured_frustum_los_area_m2=float((seen&region).sum()*.0025),
                captured_frustum_los_fraction=float((seen&region).sum()/region.sum()))
        detail=dict(coverage=coverage,coverage_note='Asset-height point visibility is a geometric upper bound, not certified marker/readability coverage',
                    acquisition_stop=acq,actual_captures=len(sensors),handoff=handoff,
                    acquisition_interruptions=[e for e in exploration if e['type'] in ['goal_interrupted','finished']],
                    method_terminal=next((e for e in reversed(events) if e['event']=='finish'),None))
        detail['search_plans']=[{k:e[k] for k in ['time','purpose','start','goal','path']} for e in events if e['event']=='plan']
        detail['observations']=[dict(id=e['id'],time=e['sensor_time'],purpose=e['purpose'],
            ids=e['vision'].get('observed_ids',[]),identity=e['vision']['identity_decision'],
            target_map=e.get('observed_asset_map'),associated_pose=e['associated_pose'],
            reading=e['vision'].get('reading',{}).get('decision')) for e in events if e['event']=='observation']
        intended=[(s,a) for s in sensors for a in s['assets'] if a['actual_id']==7]
        detail['target_usable_captures']=[s['id'] for s,a in intended if a.get('usable_image')]
        detail['target_front_surface_rendered_captures']=[s['id'] for s,a in intended if a.get('rendered')]
        detail['target_marker_decoded_captures']=[e['id'] for e in detail['observations'] if 7 in e['ids']]
        detail['wrong_marker_decoded_captures']=[e['id'] for e in detail['observations'] if 8 in e['ids']]
        for stage,file in [('acquisition','acquisition/mapper/mapper_tf_pose.csv'),('inspection','inspection/estimates.csv')]:
            if (out/file).exists():
                estimate=csv_pose(out/file)
                if len(estimate):
                    if stage=='acquisition':estimate[:,1:3]+=frame[:2]
                    detail[stage+'_localization']=error_summary(estimate,truth)
        if preflight[case]['target_present']:
            witness=preflight[case]['witness_pose']
            detail['evaluator_witness']=witness
            detail['witness_in_final_free_map']=bool(map_mask(np.array([witness[:2]]),actual_map,frame)[0][0]) if actual_map.exists() else False
            initial=next((e for e in exploration if e['type']=='frontier_decision'),None)
            if initial:
                detail['initial_decision_time']=initial['time']
                detail['witness_in_initial_free_map']=bool(map_mask(np.array([witness[:2]]),Path(initial['map_file']),frame,True)[0][0])
            if handoff and (out/'inspection/map.txt').exists():
                planner=BufferedPlanner(out/'inspection/map.txt')
                detail['oracle_witness_path_in_acquired_map']=planner.path(handoff['amcl_initial_pose'],witness)
                detail['oracle_path_note']='Offline target-coordinate query to diagnose map access; never a method input'
                detail['target_error_at_observations']=[dict(id=e['id'],error_m=float(np.linalg.norm(np.array(e['target_map']['position'])-scene['assets'][0]['pose'][:2])))
                    for e in detail['observations'] if e['target_map'] is not None]
        final=detail['method_terminal']
        if final and final.get('success_claim'):
            oid=final['final_observation']['id']
            detail['claimed_observation']=next(m for m in result['all_observations'] if m['id']==oid)
        guards={}
        for stage in ['acquisition','inspection']:
            commands=rows(out/stage/'commands.jsonl')
            guards[stage]=dict(modified=sum(e['guard'] is not None for e in commands),
                first_modified=next((e for e in commands if e['guard'] is not None),None))
        detail['guards']=guards
        tracekeys=['type','event','time','reason','pose','goal','blocked_by_new_map','progress_timeout','success_claim','purpose','commanded_goal']
        detail['compact_trace']=[{k:e[k] for k in tracekeys if k in e} for e in exploration if e['type'] in ['goal_interrupted','handoff_requested','finished']]
        detail['compact_trace'] += [{k:e[k] for k in tracekeys if k in e} for e in events if e['event'] in ['localization_ready','navigation_arrived','observation_rejected','finish']]
        diagnostics[case]=detail
        ledger.append(dict(case=case,target_present=preflight[case]['target_present'],complete_mission=result['complete_contract'],
            false_completion=result['false_completion'],wrong_asset_acceptances=result['wrong_asset_acceptances'],
            honest_noncompletion=result.get('honest_noncompletion',False),reason=result.get('inspection_reason',result['termination']),
            contacts=result['contacts'],clearance_m=result['clearance_m'],interventions=result['interventions'],resets=result['resets_after_start'],
            handoff_success=result['handoff_success'],navigation_failure=result.get('navigation_failure',False),
            acquisition_interruptions=len(acq.get('failed_goals',[])),total_sim_s=result['total_sim_s'],total_path_m=result['total_path_m'],
            acquisition_sim_s=acq.get('time'),acquisition_wheel_m=acq.get('wheel_distance_m'),acquisition_stop=acq.get('exploration_termination'),
            inspection_sim_s=result.get('inspection_time_s'),observations=len(sensors),
            actual_target_usable_captures=len(detail['target_usable_captures']),target_marker_decoded_captures=len(detail['target_marker_decoded_captures']),
            wrong_marker_decoded_captures=len(detail['wrong_marker_decoded_captures']),
            mapped_fraction=coverage['all']['mapped_fraction'],visual_upper_bound_fraction=coverage['all']['captured_frustum_los_fraction'],
            wall_s=run['wall_s'],infrastructure_failure=result['infrastructure_failure'],all_owned_processes_reaped=run['all_owned_processes_reaped']))
        plotdata[case]=dict(axis=axis,points=points,connected=connected,mapped=mapped,seen=seen,truth=truth,scene=scene,sensors=sensors,handoff=handoff)
    if ledger:
        with (outdir/'missions.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    save(outdir/'diagnostics.json',diagnostics)
    preserve=load(root/'preservation_before.json');frozen=load(root/'execution_freeze.json')
    audit=dict(protected_files=len(preserve),protected_mismatches=[p for p,h in preserve.items() if sha(ROOT/p)!=h],
        frozen_files=len(frozen['files']),frozen_mismatches=[p for p,h in frozen['files'].items() if sha(Path(p))!=h],
        attempts_started=sum(e['event']=='attempt_started' for e in rows(root/'attempts.jsonl')),completed_scored=len(ledger),
        final=not partial,all_completed_owned_processes_reaped=all(r['all_owned_processes_reaped'] for r in ledger))
    save(outdir/'integrity.json',audit)
    plot(root,plotdata,preflight)
    print(json.dumps(ledger,indent=2))


def plot(root,data,preflight):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Patch
    n=len(data)
    if not n:return
    fig,axes=plt.subplots(1,n,figsize=(4*n,5),squeeze=False)
    for ax,(case,d) in zip(axes[0],data.items()):
        color=np.full((*d['connected'].shape,3),.8)
        connected=d['connected'];mapped=d['mapped'].reshape(connected.shape);seen=d['seen'].reshape(connected.shape)
        color[connected]=[1,1,1];color[connected&mapped]=[.63,.78,1];color[connected&seen]=[1,.76,.44];color[connected&mapped&seen]=[.6,.86,.63]
        ax.imshow(color,origin='lower',extent=[-3.2,3.2,-3.2,3.2])
        for x,y,sx,sy,h in d['scene']['obstacles']:ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,color='#333333'))
        t=d['truth'];cut=d['handoff']['estimated_state']['time'] if d['handoff'] else t[-1,0]
        acq=t[t[:,0]<=cut];ins=t[t[:,0]>=cut]
        ax.plot(acq[:,1],acq[:,2],color='navy',lw=.8);ax.plot(ins[:,1],ins[:,2],color='#b000a0',lw=1)
        for s in d['sensors']:
            x,y,yaw=s['robot_pose'];ax.plot(x,y,'.',color='#b000a0',ms=3)
        for a in d['scene']['assets']:ax.plot(*a['pose'][:2],'*' if a['id']==7 else 'x',color='red' if a['id']==7 else 'black',ms=10)
        if preflight[case]['target_present']:ax.plot(*preflight[case]['witness_pose'][:2],'o',fillstyle='none',color='red',ms=5)
        ax.set(title=case,xlim=(-3.3,3.3),ylim=(-3.3,3.3),aspect='equal',xlabel='Site x (m)')
    axes[0,0].set_ylabel('Site y (m)')
    fig.legend(handles=[Patch(color=[.63,.78,1],label='mapped free only'),Patch(color=[1,.76,.44],label='camera LOS only'),Patch(color=[.6,.86,.63],label='both'),Patch(color='white',label='neither')],loc='lower center',ncol=4)
    fig.suptitle('Evaluator overlays: acquired-map coverage versus actual capture frusta\nBlue path: acquisition; magenta: inspection. Stars: intended asset; cross: ID8; circle: oracle feasible view.',fontsize=11)
    fig.tight_layout(rect=[0,.07,1,.89]);fig.savefig(root/'analysis/coverage_and_paths.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args();analyze(a.root.resolve(),a.partial)
