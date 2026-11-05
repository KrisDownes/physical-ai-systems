"""Post-run analysis/demo only: explicitly reads evaluator truth, never controls a robot."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/site_moved_asset_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def load(path): return json.loads(path.read_text())
def rows(path): return [json.loads(s) for s in path.read_text().splitlines()]
def save(path, data): path.write_text(json.dumps(data, indent=2)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def wrap(x): return math.atan2(math.sin(x), math.cos(x))


def analyze(root, output, video=False):
    output.mkdir(parents=True, exist_ok=False)
    freeze=load(root/'freeze.json'); table=[]; diagnostics=[]; datasets={}
    for job in freeze['jobs']:
        folder=Path(job['output']); result=load(folder/'outcome.json')
        events=rows(folder/'method.jsonl'); sensor=rows(folder/'private/sensor_truth.jsonl')
        truth=[r for r in rows(folder/'judge.jsonl') if r['type']=='truth']
        trajectory=np.array([[r['time'],r['x'],r['y']] for r in truth])
        run=load(folder/'run.json')
        table.append(dict(mission=job['name'],mode=job['mode'],case=job['case'],
            completed=result['correct_physical_task_completion'],claim=result['success_claim'],
            false_claim=result['false_completion_claim'],wrong_asset_acceptances=result['wrong_asset_acceptances'],
            simulation_s=result['simulation_time_s'],distance_m=result['physical_path_m'],
            captures=result['captures'],recovery_actions=result['recovery_actions'],
            inspection_abstentions=result['inspection_abstentions'],
            first_detection_s=result['first_target_detection_sim_s'],
            first_detection_distance_m=result['distance_to_first_target_detection_m'],
            contacts=result['contact_events'],clearance_m=result['minimum_clearance_m'],
            navigation_failure=result['navigation_failure'],infrastructure_failure=result['infrastructure_failure'],
            reason=result['termination_reason'],wall_s=run['wall_s'],
            children_peak_rss_kib=run['children_peak_rss_kib'],all_children_reaped=run['all_owned_processes_reaped']))
        for e in events:
            if e['event']!='observation' or not e.get('observed_asset_map'): continue
            s=next(s for s in sensor if s['id']==e['id'])
            asset=next(a for a in s['assets'] if a['actual_id']==7)
            true_robot=np.array(s['robot_pose']); estimated_robot=np.array(e['associated_pose'])
            true_asset=np.array(asset['asset_pose'][:2]); observed=np.array(e['observed_asset_map']['position'])
            c,z=e['vision']['target']['gauge_camera'][0],e['vision']['target']['gauge_camera'][2]
            yaw=true_robot[2];rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            # Diagnostic oracle: substitute true robot pose, leaving pixel-derived relative pose fixed.
            oracle_asset=true_robot[:2]+rot@np.array([z,-c])
            diagnostics.append(dict(mission=job['name'],capture=e['id'],stamp=e['sensor_time'],
                associated_robot=e['associated_pose'],true_robot=s['robot_pose'],
                pixel_gauge_camera=e['vision']['target']['gauge_camera'],
                observed_asset=observed.tolist(),true_asset=true_asset.tolist(),
                map_asset_error_m=float(np.linalg.norm(observed-true_asset)),
                pixel_relative_error_with_oracle_robot_m=float(np.linalg.norm(oracle_asset-true_asset)),
                robot_localization_position_error_m=float(np.linalg.norm(estimated_robot[:2]-true_robot[:2])),
                robot_localization_heading_error_deg=math.degrees(wrap(estimated_robot[2]-true_robot[2])),
                processing_age_s=e['decision_time']-e['sensor_time'],
                reprojection_rms_px=e['vision']['target']['rms_px']))
        datasets[job['name']]=(folder,events,sensor,trajectory,result)
    with (output/'missions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    save(output/'transform_diagnostics.json',diagnostics)
    project=Path(__file__).resolve().parents[2]
    before=load(root.parent/'preservation_before.json')
    mismatches=[p for p,h in before.items() if sha(project/p)!=h]
    frozen_mismatches=[p for p,h in freeze['files'].items() if sha(Path(p))!=h]
    matched={}
    for case in ['unchanged','moved','wrong_asset','absent']:
        a=root/(case+'__old_location');b=root/(case+'__autonomous')
        filenames=['map.txt','controller_config.json','public/vision.json','private/scene.json','private/evaluation.json','private/world.sdf']
        aa,bb=load(a/'public/policy.json'),load(b/'public/policy.json');aa.pop('mode');bb.pop('mode')
        matched[case]=dict(files={n:sha(a/n)==sha(b/n) for n in filenames},policy_except_mode=aa==bb)
    save(output/'audit.json',dict(freeze_sha256=sha(root/'freeze.json'),protected_files=len(before),
        protected_mismatches=mismatches,frozen_mismatches=frozen_mismatches,matched_inputs=matched,
        launches=len([r for r in rows(root/'attempts.jsonl') if r['event']=='start']),
        finished=len([r for r in rows(root/'attempts.jsonl') if r['event']=='finish'])))

    def map_plot(ax,folder,trajectory,end=None):
        scene=load(folder/'private/scene.json')
        for ox,oy,sx,sy,_ in scene['obstacles']:
            ax.add_patch(Rectangle((ox-sx/2,oy-sy/2),sx,sy,color='#777777'))
        for a in scene['assets']:
            x,y,_=a['pose'];ax.scatter(x,y,c='#b32237' if a['id']==7 else '#d08400',marker='s',s=65)
            ax.text(x+.08,y+.08,f"ID{a['id']} (judge)",fontsize=8)
        ax.scatter(-1.5,-1.5,c='black',marker='x',s=40);ax.text(-2.9,-1.7,'old view',fontsize=8)
        used=trajectory if end is None else trajectory[trajectory[:,0]<=end]
        if not len(used): used=trajectory[:1]
        ax.plot(used[:,1],used[:,2],color='#116da7',lw=2)
        ax.plot(used[-1,1],used[-1,2],'o',color='#116da7')
        ax.set(xlim=(-3.4,3.4),ylim=(-3.4,3.4),xlabel='map x (m)',ylabel='map y (m)')
        ax.set_aspect('equal');ax.grid(alpha=.15)

    fig,axes=plt.subplots(2,2,figsize=(10,10))
    for ax,case in zip(axes.flat,['unchanged','moved','wrong_asset','absent']):
        folder,events,sensor,trajectory,result=datasets[case+'__autonomous']
        map_plot(ax,folder,trajectory)
        ax.set_title(f"{case}: {'completed' if result['correct_physical_task_completion'] else result['termination_reason']}\n"
                     f"{result['physical_path_m']:.2f}m / {result['simulation_time_s']:.1f}s / {result['captures']} observations",fontsize=10)
    fig.suptitle('Autonomous development missions — physical trajectories and evaluator asset locations')
    fig.tight_layout();fig.savefig(output/'trajectories.png',dpi=140);plt.close(fig)

    folder,events,sensor,trajectory,result=datasets['wrong_asset__autonomous']
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,number,label in zip(axes,[0,16,18],['Old view: ID8, continue search','Search: ID7, confirm pose','Approach: fresh ID7 + readable dial']):
        im=plt.imread(folder/'camera'/f'observation_{number:03d}.png');ax.imshow(im,cmap='gray',vmin=0,vmax=1)
        v=load(folder/f'vision_{number:03d}.json')
        for d in v.get('detections',[]):
            uv=np.array(d['corners']);uv=np.vstack([uv,uv[0]])
            ax.plot(uv[:,0],uv[:,1],color='#00b8ff');ax.text(uv[0,0],uv[0,1]-20,f"detected ID{d['id']}",color='#00b8ff')
        ax.set_title(label,fontsize=11);ax.set_xlim(200,1800);ax.set_ylim(900,150);ax.axis('off')
    fig.suptitle('Recorded synthetic camera pixels; blue annotations are detector outputs')
    fig.tight_layout();fig.savefig(output/'observation_sequence.png',dpi=150);plt.close(fig)
    if video:
        from matplotlib.animation import FFMpegWriter
        writer=FFMpegWriter(fps=4,metadata={'title':'Recorded moved-asset development mission'})
        fig=plt.figure(figsize=(12,5))
        with writer.saving(fig,str(output/'demo.mp4'),100):
            for seconds in np.linspace(0,result['simulation_time_s'],100):
                fig.clear();left,right=fig.subplots(1,2);map_plot(left,folder,trajectory,seconds)
                left.set_title(f'Wrong-asset case: {seconds:.1f} / {result["simulation_time_s"]:.1f}s\nEvaluator view; asset positions hidden from policy')
                available=[s for s in sensor if s['time']<=seconds]
                if available:
                    s=available[-1];v=load(folder/f'vision_{s["id"]:03d}.json')
                    right.imshow(plt.imread(folder/'camera'/s['file']),cmap='gray')
                    right.set_title(f"Latest captured image @ {s['time']:.1f}s\nIDs {v.get('observed_ids',[])}; reading {v.get('reading',{}).get('decision','not requested')}")
                else:right.text(.1,.5,'No image captured yet')
                right.axis('off');fig.tight_layout();writer.grab_frame()
        plt.close(fig)
    print(json.dumps(dict(missions=len(table),protected_mismatches=mismatches,frozen_mismatches=frozen_mismatches,output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args()
    analyze(a.runs,a.out,a.video)
