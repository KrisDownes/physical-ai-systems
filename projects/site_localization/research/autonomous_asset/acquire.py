"""Adapt the existing sensor-driven SLAM runner to a commissioned cell survey.

World and evaluator geometry are setup-only. Controller input contains no changed
geometry. Its diagnostic AMCL receives a featureless grid, never the oracle map.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
import acquisition_run as run


def fixture(out,template,commissioning):
    cfg=json.loads(commissioning.read_text())
    # Only the independent legacy acquisition judge gets obstacle geometry.
    evaluation=json.loads((template/'private/evaluation.json').read_text())
    cfg['obstacles']=evaluation['physical_obstacles']
    cfg.update(condition=template.name,scenario='assisted_post_change_survey',remedy='acquired_map',
        map_from_mission=[2.,2.,0.],physical_target_frame='commissioned site',
        initialization='Known repeatable dock, fixed initial SLAM node; no later truth',
        operator_events=[dict(type='commissioned_route',count=1,human_seconds=None),
                         dict(type='known_dock_initialization',count=1,human_seconds=None)],
        reset_count=1,repeat_attempt_index=0,max_sim_s=240,
        dock_completion=dict(pose=cfg['start'],position_tolerance_m=.15,
            yaw_tolerance_rad=.17453292519943295,stationary_dwell_s=2))
    cfg['route']=[dict(pose=g,inspection_index=i) for i,g in enumerate(cfg['goals'])]
    cfg['route'] += [dict(pose=[-1.5,-1.5,0],inspection_index=None),dict(pose=cfg['start'],inspection_index=None)]
    shutil.copyfile(template/'private/world.sdf',out/'cell.sdf')
    # Existing Mission always starts diagnostic AMCL, even with SLAM controlling.
    # An arbitrary free grid supplies no scene information and is not used to steer.
    (out/'map.txt').write_text('40 40 0.5 -10 -10\n'+(' '.join(['-1']*40)+'\n')*40)
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True)
    p.add_argument('--explore',action='store_true');p.add_argument('--commissioning',type=Path,required=True);a,rest=p.parse_known_args()
    def make(out,condition,remedy):
        cfg=fixture(out,a.template,a.commissioning)
        if a.explore:
            cfg.update(goals=[cfg['start']],route=[dict(pose=cfg['start'],inspection_index=None)],scenario='frontier_acquisition')
            cfg['operator_events']=[dict(type='known_dock_initialization',count=1,human_seconds=None)]
        return cfg
    run.make=make
    if a.explore:rest.append('--explore')
    sys.argv=['existing_sensor_mapping_runner',*rest,'--condition','unchanged','--mapping','new',
        '--localizer','slam_toolbox','--map-from-mission','2','2','0','--arrival-tolerance','.03',
        '--startup-readiness','--slam-pose-source','tf']
    run.main()
