"""Research development runner; old sim/ baseline and tutor snapshot untouched."""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import uuid
import yaml
from scenarios import make

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition',choices=['unchanged','frame_shift','physical_change'],required=True)
    p.add_argument('--layout',choices=['cell','two_zone'],default='cell')
    p.add_argument('--remedy',choices=['fixed','metadata_alignment','acquired_map'],default='fixed')
    p.add_argument('--map',type=Path)
    p.add_argument('--map-from-mission',type=float,nargs=3)
    p.add_argument('--localizer',choices=['amcl','survey_odometry','slam_toolbox'],default='amcl')
    p.add_argument('--mapping',choices=['off','new','continued','localization'],default='off')
    p.add_argument('--session',type=Path)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--domain',type=int,default=86)
    p.add_argument('--wall-limit',type=float,default=200)
    p.add_argument('--arrival-tolerance',type=float,default=.09)
    p.add_argument('--startup-readiness',action='store_true')
    p.add_argument('--slam-pose-source',choices=['pose','tf'],default='pose')
    p.add_argument('--stop-after-goals',type=int)
    p.add_argument('--return-inspect',action='store_true',help='Six inspections followed by independently judged dock return (two_zone only)')
    a=p.parse_args()
    if not 0<a.arrival_tolerance<=.09:p.error('Arrival tolerance must be in (0,.09] metres')
    if not 0<=a.domain<=232:p.error('ROS domain must be between 0 and 232')
    domain_lock=open(f'/tmp/site_localization_domain_{a.domain}.lock','a+')
    try:fcntl.flock(domain_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError(f'Research ROS domain {a.domain} is already in use')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.layout=='two_zone':
        from two_zone import make as generate_two_zone
        cfg=generate_two_zone(out,a.condition,a.remedy)
    else:cfg=make(out,a.condition,a.remedy)
    if a.return_inspect:
        if a.layout!='two_zone':p.error('--return-inspect requires two_zone')
        from two_zone import extend_return_loop
        extend_return_loop(cfg)
    cfg['control_dwell_margin_s']=.2
    cfg['arrival_tolerance_m']=a.arrival_tolerance
    cfg['slam_pose_source']=a.slam_pose_source
    if a.map:
        (out/'map.txt').write_bytes(a.map.read_bytes());cfg['map_source']=str(a.map.resolve())
    if a.map_from_mission is not None:cfg['map_from_mission']=a.map_from_mission
    cfg.update(localizer=a.localizer,seed=a.seed,mapping=a.mapping)
    if a.stop_after_goals:cfg['stop_after_goals']=a.stop_after_goals
    if a.localizer=='survey_odometry':
        cfg['operator_events'].append(dict(type='scripted_survey_proxy',count=1,human_seconds=None))
    if a.mapping!='off' and a.localizer=='slam_toolbox' and a.map_from_mission is None:
        raise ValueError('Declare mapper frame relative to fixed mission frame')
    if a.mapping in ['continued','localization'] and a.session is None:raise ValueError('Serialized graph required')
    (out/'cell.json').write_text(json.dumps(cfg,indent=2)+'\n')
    # Physically separate evaluator geometry from controller input. Targets,
    # calibration/assistance and route are commissioning information; changed
    # obstacle boxes and simulator predicates are never method inputs.
    controller_keys=['map_from_mission','route','start','localizer','goals','seed',
        'sensor_model','dwell_s','stop_after_goals','control_dwell_margin_s','dock_completion','arrival_tolerance_m','slam_pose_source']
    (out/'controller_config.json').write_text(json.dumps({k:cfg[k] for k in controller_keys if k in cfg},indent=2)+'\n')
    env=os.environ.copy();env.update(GZ_PARTITION='site_research_'+uuid.uuid4().hex,
        ROS_DOMAIN_ID=str(a.domain),ROS_LOG_DIR=str(out/'ros_logs'),GZ_LOG_PATH=str(out/'gz_logs'),
        LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',EGL_PLATFORM='surfaceless')
    bridges=['/site/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/site/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/site/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/judge/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/site/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
        '/judge/wheel_contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    commands=dict(bridge=['/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge',*bridges],
        judge=['/usr/bin/python3',str(ROOT/'sim/judge_node.py'),str(out)])
    if a.mapping!='off':
        params=yaml.safe_load((ROOT/'research/slam_params.yaml').read_text());sp=params['slam_toolbox']['ros__parameters']
        if a.session:sp.update(map_file_name=str(a.session.resolve()),map_start_pose=[0.,0.,0.])
        if a.mapping=='localization':sp['mode']='localization'
        (out/'slam_params.yaml').write_text(yaml.safe_dump(params))
        launch='localization_launch.py' if a.mapping=='localization' else 'online_sync_launch.py'
        commands['mapper']=['ros2','launch','slam_toolbox',launch,
            'slam_params_file:='+str(out/'slam_params.yaml'),'use_sim_time:=true']
        commands['mapping_bridge']=['/usr/bin/python3',str(ROOT/'research/slam_bridge.py'),
            '--out',str(out/'mapper'),'--ros-args','-p','use_sim_time:=true']
    commands['controller']=['/usr/bin/python3',str(ROOT/'research/controller.py'),str(out)]
    commands['server']=['gz','sim','-s','-r','--headless-rendering','-v','3',str(out/'cell.sdf')]
    files=[ROOT/'research'/f for f in ['run.py','controller.py','frames.py','scenarios.py','score.py']]
    if a.layout=='two_zone':files.append(ROOT/'research/two_zone.py')
    files += [ROOT/'sim/judge_node.py',ROOT/'sim/score_mission.py',ROOT/'vendor/offline_amcl.cpp',ROOT/'vendor/offline_amcl']
    if a.mapping!='off':files += [ROOT/'research/slam_bridge.py',out/'slam_params.yaml']
    if a.startup_readiness:files += [ROOT/'research/readiness.py']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    snapshot=out/'source';snapshot.mkdir()
    for f in files:
        if f.suffix in ['.py','.yaml','.cpp']:(snapshot/f.name).write_bytes(f.read_bytes())
    children={};handles=[];started=time.monotonic();reason='wall_guard';finished=None;serialization=None
    try:
        for name,cmd in commands.items():
            if name=='server' and a.startup_readiness and a.localizer=='slam_toolbox':
                try:
                    reply=subprocess.run(['/usr/bin/python3',str(ROOT/'research/readiness.py'),
                        '--output',str(out/'startup_readiness.json')],env=env,capture_output=True,text=True,timeout=45)
                except subprocess.TimeoutExpired:
                    (out/'startup_readiness.json').write_text(json.dumps(dict(ready=False,
                        error='45 second helper timeout',physics_started=False),indent=2)+'\n')
                    reason='readiness_failed';break
                (out/'startup_readiness.stderr').write_text(reply.stderr)
                if reply.returncode:
                    reason='readiness_failed';break
            stdout=(out/f'{name}.stdout').open('w');stderr=(out/f'{name}.stderr').open('w');handles.extend([stdout,stderr])
            children[name]=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            if name=='controller':time.sleep(1)
        while 'server' in children and time.monotonic()-started<a.wall_limit:
            if any(proc.poll() is not None for proc in children.values()):reason='child_exit';break
            if (out/'controller_done.json').exists():
                if finished is None:finished=time.monotonic()
                if time.monotonic()-finished>3:reason='controller_completed';break
            time.sleep(.2)
        if 'server' in children and a.mapping in ['new','continued'] and children['mapper'].poll() is None:
            command=['ros2','service','call','/slam_toolbox/serialize_map','slam_toolbox/srv/SerializePoseGraph',
                     '{filename: "'+str(out/'mapper/session')+'"}']
            before=time.monotonic()
            try:
                reply=subprocess.run(command,env=env,capture_output=True,text=True,timeout=8)
                serialization=dict(returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr,wall_s=time.monotonic()-before)
            except subprocess.TimeoutExpired:serialization=dict(error='timeout',wall_s=time.monotonic()-before)
    except KeyboardInterrupt:
        reason='operator_abort'
    finally:
        for proc in children.values():
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGINT)
                except ProcessLookupError:pass
        for proc in children.values():
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for f in handles:f.close()
    record=dict(stop_reason=reason,wall_s=time.monotonic()-started,commands=commands,
        physics_started='server' in children,startup_readiness_requested=a.startup_readiness,
        exits={k:v.returncode for k,v in children.items()},partition=env['GZ_PARTITION'],ros_domain=a.domain,
        serialization=serialization,reset_count=1,repeat_attempt_index=0,
        all_owned_processes_reaped=all(p.poll() is not None for p in children.values()),
        children_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        children_cpu_user_s=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        children_cpu_system_s=resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    reply=subprocess.run(['/usr/bin/python3',str(ROOT/'research/score.py'),str(out)],capture_output=True,text=True)
    (out/'score.stdout').write_text(reply.stdout);(out/'score.stderr').write_text(reply.stderr)
    record['score_returncode']=reply.returncode
    record['evaluator_sha256_at_scoring']=hashlib.sha256((ROOT/'research/score.py').read_bytes()).hexdigest()
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    if reply.stdout:print(reply.stdout)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
