"""Bounded six-condition bottle experiment; all condition data stay in evaluator."""
import argparse,json,time
from pathlib import Path
import numpy as np
from bottle_task import BottleEnvironment,physical_parameters
from environment import Q_HOME
from nominal import run
from run_store import create_run
from diagnose import overlay

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=create_run('bottle',cfg);(out/'episodes').mkdir();rows=[]
    for cond in cfg['conditions']:
        for case in cfg['cases']:
            e=BottleEnvironment(cond['fill'],cond['grip'],physics_timestep=cfg.get('physics_timestep'),action_budget=cfg['action_budget'],movement_budget=cfg['movement_budget_rad'])
            q=Q_HOME.copy();q[0]=case['initial_pan'];start=time.perf_counter()
            try:r=run(e,case['object_xy'],q,method='calibration',seed=case['seed'],diagnostics=(case==cfg['cases'][0] and cfg['record_videos']),grasp_offset=cfg.get('grasp_offset'),close_ramp_actions=cfg.get('close_ramp_actions',0),transport_speed=cfg.get('transport_speed',1.),grip_closed=cfg.get('grip_closed',-.1))
            except RuntimeError as error:
                if 'budget exhausted' not in str(error):raise
                r={'success':False,'failure':'budget','error':str(error)}
            frames=r.pop('diagnostic_frames',[]);r.pop('frames',None)
            if frames:overlay(frames,[0,0],out/(cond['fill']+'_'+cond['grip']+'.gif'))
            ev=r.get('evaluation',{});snap=ev.get('snapshots',{});final=snap.get('after_settle')
            if not r['success'] and r.get('failure') is None:r['failure']='stability_or_tilt'
            if final: placement=float(np.linalg.norm(np.asarray(final['object_xyz'][:2])-[.22,-.05]))
            else:placement=None
            row={'condition':cond,'case':case,'model':physical_parameters(cond['fill']),'result':r,'actions':e.action_count,
              'movement_rad':e.movement,'sim_s':float(e.data.time),'wall_s':time.perf_counter()-start,'compute_s':e.compute_s,
              'retries':0,'placement_error_m':placement,'telemetry_evaluator':e.telemetry,'joint_limit_physics_steps':e.physics_limit_steps,
              'self_collision_physics_steps':e.physics_collision_steps,'self_collision_pairs':e.physics_collision_pairs,
              'command_saturation_actions':e.saturated_actions,'trace':e.trace}
            with (out/'episodes'/f"{cond['fill']}_{cond['grip']}_{case['id']}.json").open('x') as f:json.dump(row,f,indent=2)
            rows.append(row);e.close();print(cond['fill'],cond['grip'],case['id'],r['success'],r.get('failure'),flush=True)
    summary={}
    for cond in cfg['conditions']:
        group=[r for r in rows if r['condition']==cond];name=cond['fill']+'_'+cond['grip']
        summary[name]={'successes':sum(x['result']['success'] for x in group),'trials':len(group),
          'failures':[x['result'].get('failure') for x in group if not x['result']['success']],
          'grasp_acquired':sum(x['result'].get('evaluation',{}).get('grasp_contact',False) for x in group),
          'lifted':sum(x['result'].get('evaluation',{}).get('lifted',False) for x in group),
          'median_actions':float(np.median([x['actions'] for x in group])),
          'median_sim_s':float(np.median([x['sim_s'] for x in group])),
          'median_wall_s':float(np.median([x['wall_s'] for x in group])),
          'median_movement_rad':float(np.median([x['movement_rad'] for x in group])),
          'median_compute_s':float(np.median([x['compute_s'] for x in group])),
          'max_slip_m':max(x['telemetry_evaluator']['max_grip_relative_motion_m'] for x in group),
          'max_tilt_deg':max(x['telemetry_evaluator']['max_tilt_deg'] for x in group),
          'max_contact_normal_N':max(x['telemetry_evaluator']['peak_normal_N'] for x in group),
          'physics_self_collision_steps':sum(x['self_collision_physics_steps'] for x in group),
          'physics_joint_limit_steps':sum(x['joint_limit_physics_steps'] for x in group)}
    with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
    print('saved',out);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
