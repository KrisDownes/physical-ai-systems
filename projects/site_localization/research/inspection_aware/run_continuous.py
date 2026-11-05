"""One Gazebo lifetime. Sensor-only map/estimated-pose handoff, no teleport/reset."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent.parent/'continuous_asset';ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research'))
from frames import compose
from correction_pipeline import anchor

def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def no_server():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        except OSError:continue
        if any(Path(x).name=='gz' and args[i+1:i+3]==['sim','-s'] for i,x in enumerate(args)):
            raise RuntimeError('Other simulator active: '+str(p))

def main(a):
    no_server();lock=open('/tmp/site_localization_domain_216.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);acq=out/'acquisition';acq.mkdir()
    shutil.copytree(a.template/'private',out/'private');shutil.copytree(a.template/'public',out/'public')
    cfg=dict(start=[-2.,-2.,0.],goals=[[-2.,-2.,0.]],route=[dict(pose=[-2.,-2.,0.],inspection_index=None)],
        map_from_mission=[2.,2.,0.],localizer='slam_toolbox',seed=a.seed,dwell_s=2,
        control_dwell_margin_s=.2,arrival_tolerance_m=.03,slam_pose_source='tf')
    save(acq/'controller_config.json',cfg);save(acq/'exploration_config.json',dict(require_return=a.return_only))
    (acq/'map.txt').write_text('40 40 0.5 -10 -10\n'+(' '.join(['-1']*40)+'\n')*40)
    slam_source=a.slam_params if a.slam_params is not None else ROOT/'research/slam_params.yaml'
    shutil.copyfile(slam_source,out/'slam_params.yaml')
    files=[Path(__file__).resolve()]+[p for p in HERE.glob('*.py')]+[ROOT/'docs/continuous_asset/protocol.md',acq/'controller_config.json',acq/'exploration_config.json',out/'slam_params.yaml']
    files += [p for p in (ROOT/'research/autonomous_asset/reused_rover').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    files += [ROOT/'research/autonomous_asset'/n for n in ['pose.py','vision.py']]
    files += [ROOT/'research'/n for n in ['frames.py','slam_bridge.py','slam_params.yaml','readiness.py','correction_pipeline.py']]
    if a.slam_params is not None:files.append(a.slam_params)
    files += [ROOT/'research/moved_asset'/n for n in ['planning.py','camera.py','render.py']]
    files += [ROOT/'sim/judge_node.py',ROOT/'vendor/offline_amcl',ROOT/'vendor/offline_amcl.cpp']
    files += [p for p in (ROOT/'results/asset_calibration_20261023/frozen_bundle').iterdir() if p.is_file()]
    files += [p for p in a.template.rglob('*') if p.is_file()]
    save(out/'freeze.json',dict(revision='continuous-v2-with-explicit-slam-config',utc=now(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files={str(p):sha(p) for p in files},slam_params_source=str(slam_source),return_only=a.return_only,seed=a.seed,domain=216))
    (out/'source').mkdir()
    for p in HERE.glob('*.py'):shutil.copyfile(p,out/'source'/p.name)
    shutil.copyfile(Path(__file__).resolve(),out/'source'/'inspection_aware_run_continuous.py')
    env=os.environ.copy();env.update(ROS_DOMAIN_ID='216',GZ_PARTITION='continuous_'+uuid.uuid4().hex,
        ROS_LOG_DIR=str(out/'ros_logs'),GZ_LOG_PATH=str(out/'gz_logs'),LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',EGL_PLATFORM='surfaceless')
    bridges=['/site/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist','/site/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/site/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan','/judge/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        '/site/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts','/judge/wheel_contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts','/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    children={};expected_stopped=set();handles=[];commands={};started=time.monotonic();reason='wall_guard';stage='acquisition';handoff=None
    ledger=(out.parent/'attempts.jsonl').open('a')
    def event(kind,**d):
        value=dict(event=kind,utc=now(),output=str(out),**d);ledger.write(json.dumps(value)+'\n');ledger.flush()
        with (out/'stages.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    def launch(name,cmd):
        commands[name]=cmd;fs=[(out/(name+'.stdout')).open('w'),(out/(name+'.stderr')).open('w')];handles.extend(fs)
        children[name]=subprocess.Popen(cmd,env=env,stdout=fs[0],stderr=fs[1],start_new_session=True)
        event('process_started',name=name,pid=children[name].pid,command=cmd)
    def stop(name):
        proc=children[name];expected_stopped.add(name)
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    event('attempt_started',template=str(a.template),return_only=a.return_only,initializations=1,resets_after_start=0)
    try:
        launch('bridge',['/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge',*bridges])
        launch('judge',['/usr/bin/python3',str(ROOT/'sim/judge_node.py'),str(out)])
        launch('mapper',['ros2','launch','slam_toolbox','online_sync_launch.py','slam_params_file:='+str(out/'slam_params.yaml'),'use_sim_time:=true'])
        launch('mapping_bridge',['/usr/bin/python3',str(ROOT/'research/slam_bridge.py'),'--out',str(acq/'mapper'),'--ros-args','-p','use_sim_time:=true'])
        launch('explorer',['/usr/bin/python3',str(HERE/'explore.py'),str(acq)])
        reply=subprocess.run(['/usr/bin/python3',str(ROOT/'research/readiness.py'),'--output',str(out/'startup_readiness.json')],env=env,capture_output=True,text=True,timeout=45)
        if reply.returncode:raise RuntimeError('startup_readiness_failed')
        launch('server',['gz','sim','-s','-r','--headless-rendering','-v','3',str(out/'private/world.sdf')])
        while time.monotonic()-started<a.wall_limit:
            exited=[n for n,p in children.items() if n not in expected_stopped and p.poll() is not None]
            if exited:reason='child_exit:'+','.join(exited);break
            if stage=='acquisition' and (acq/'handoff_ready.json').exists():
                done=load(acq/'acquisition_done.json');state=load(acq/'handoff_ready.json')
                event('acquisition_terminal',method=done,estimated_state=state)
                if a.return_only:reason='return_validation_finished';break
                if done['reason']!='mapping_endpoint_ready':reason='acquisition_failed:'+done['reason'];break
                stop('explorer')
                serialized=subprocess.run(['ros2','service','call','/slam_toolbox/serialize_map','slam_toolbox/srv/SerializePoseGraph','{filename: "'+str(acq/'mapper/session')+'"}'],env=env,capture_output=True,text=True,timeout=10)
                save(out/'serialization.json',dict(returncode=serialized.returncode,stdout=serialized.stdout,stderr=serialized.stderr))
                if serialized.returncode or 'result=0' not in serialized.stdout:raise RuntimeError('mapper_serialization_failed')
                stop('mapper');stop('mapping_bridge')
                ready_state=dict(state)
                with (acq/'mapper/mapper_tf_pose.csv').open() as f:final_tf=list(csv.DictReader(f))[-1]
                with (acq/'mapper/wheel_odometry.csv').open() as f:wheel_rows=list(csv.DictReader(f))
                stamp=float(final_tf['timestamp']);wheel=min(wheel_rows,key=lambda r:abs(float(r['timestamp'])-stamp))
                if abs(float(wheel['timestamp'])-stamp)>.1:raise RuntimeError('handoff_wheel_timestamp_missing')
                state.update(time=stamp,pose_time=stamp,map_pose=[float(final_tf[k]) for k in ['x','y','yaw']],wheel_time=float(wheel['timestamp']),wheel_pose=[float(wheel[k]) for k in ['x','y','yaw']],source='Final stopped-mapper exact-time TF; unchanged odometry')
                frame=anchor(acq,[-2.,-2.,0.])['target_from_source']
                if abs(frame[2])>1e-10:raise RuntimeError('rotated_raster_handoff_unsupported')
                ins=out/'inspection';ins.mkdir();shutil.copytree(out/'public',ins/'public');shutil.copytree(out/'private',ins/'private')
                with (acq/'mapper/map.txt').open() as f:head=np.fromstring(f.readline(),sep=' ');body=f.read()
                head[3:5]+=frame[:2];(ins/'map.txt').write_text(f'{int(head[0])} {int(head[1])} {head[2]:.12g} {head[3]:.12g} {head[4]:.12g}\n'+body)
                pose=compose(frame,state['map_pose']).tolist();cov=state['slam_covariance']
                std_xy=max(.15,math.sqrt(max(0,cov[0]+cov[7])));std_yaw=max(10.,math.degrees(math.sqrt(max(0,cov[35]))))
                if std_xy>.5 or std_yaw>30:raise RuntimeError('handoff_uncertainty_too_large')
                cfg=load(a.template/'controller_config.json');cfg.update(start=pose,localizer='amcl',seed=a.seed+1,initial_std_xy=std_xy,initial_std_yaw_deg=std_yaw)
                save(ins/'controller_config.json',cfg)
                settings=load(ins/'public/policy.json');settings.update(handoff_time=state['time'],acquisition_distance_m=done['wheel_distance_m'],max_sim_s=min(650,state['time']+420))
                save(ins/'public/policy.json',settings)
                handoff=dict(estimated_state=state,stationary_ready_state=ready_state,site_from_slam=frame,amcl_initial_pose=pose,amcl_initial_std_xy_m=std_xy,amcl_initial_std_yaw_deg=std_yaw,
                    map_sha256=sha(ins/'map.txt'),source_map_sha256=sha(acq/'mapper/map.txt'),map_metadata=load(acq/'mapper/map_metadata.json'),
                    odometry_reset=False,simulator_restarted=False,ground_truth_used=False,return_required=False,initial_dock_only=True,
                    uncertainty_note='Conservative diagonal seed floors; not calibrated joint uncertainty; AMCL shares map/lidar with SLAM')
                save(out/'handoff.json',handoff);event('handoff_configured',handoff=handoff,server_pid=children['server'].pid)
                launch('camera',['/usr/bin/python3',str(ROOT/'research/moved_asset/camera.py'),str(ins)])
                launch('inspector',['/usr/bin/python3',str(a.inspector_script or HERE/'inspection.py'),str(ins)])
                stage='inspection'
            if stage=='inspection' and (out/'inspection/task_done.json').exists():
                reason='inspection_finished';time.sleep(2);break
            time.sleep(.1)
    except KeyboardInterrupt:reason='operator_abort'
    except Exception as e:
        reason='infrastructure_exception:'+str(e);save(out/'exception.json',dict(type=type(e).__name__,message=str(e)))
    finally:
        for name in ['inspector','explorer','server','camera','mapper','mapping_bridge','judge','bridge']:
            if name in children:stop(name)
        for f in handles:f.close()
        run=dict(stop_reason=reason,wall_s=time.monotonic()-started,commands=commands,server_pid=children['server'].pid if 'server' in children else None,
            simulator_launches=int('server' in children),initializations=1,resets_after_start=0,interventions=int(reason=='operator_abort'),
            all_owned_processes_reaped=all(p.poll() is not None for p in children.values()),exits={n:p.returncode for n,p in children.items()},partition=env['GZ_PARTITION'],return_only=a.return_only)
        save(out/'run.json',run);event('attempt_finished',run=run);ledger.close()
    reply=subprocess.run(['/usr/bin/python3',str(HERE/'score.py'),str(out)],capture_output=True,text=True)
    (out/'score.stdout').write_text(reply.stdout);(out/'score.stderr').write_text(reply.stderr)
    print(out.name,reason,'score',reply.returncode,flush=True)
    if reply.returncode:print(reply.stderr);raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--slam-params',type=Path);p.add_argument('--return-only',action='store_true');p.add_argument('--inspector-script',type=Path);p.add_argument('--seed',type=int,default=7301);p.add_argument('--wall-limit',type=float,default=800);a=p.parse_args();a.template=a.template.resolve();main(a)
