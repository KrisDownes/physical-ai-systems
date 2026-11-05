"""Bounded isolated ROS2/Gazebo development mission; reap only owned groups."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import resource
import hashlib
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--wall-limit',type=float,default=240)
    p.add_argument('--world',type=Path)
    p.add_argument('--candidate-map',type=Path)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent;env=os.environ.copy()
    for name in ['cell.json','cell.sdf','map.txt']:shutil.copy2(root/name,out/name)
    if a.world:
        shutil.copy2(a.world,out/'cell.sdf')
        cfg=json.loads((out/'cell.json').read_text());cfg['scenario']='contact_positive_control'
        (out/'cell.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if a.candidate_map:
        shutil.copy2(a.candidate_map,out/'map.txt')
        cfg=json.loads((out/'cell.json').read_text());cfg['scenario']='shifted_map_frame_negative_control'
        cfg['candidate_map']=str(a.candidate_map.resolve())
        (out/'cell.json').write_text(json.dumps(cfg,indent=2)+'\n')
    sources=[root/f for f in ['mission_node.py','judge_node.py','score_mission.py','run_mission.py']]
    sources += [root.parent/'vendor/offline_amcl.cpp',root.parent/'vendor/offline_amcl']
    hashes={str(f.relative_to(root.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    env.update(GZ_PARTITION='site_mission_'+uuid.uuid4().hex,ROS_DOMAIN_ID='84',
               ROS_LOG_DIR=str(out/'ros_logs'),GZ_LOG_PATH=str(out/'gz_logs'),
               LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',EGL_PLATFORM='surfaceless')
    bridges=['/site/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
             '/site/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
             '/site/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
             '/judge/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry',
             '/site/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
             '/judge/wheel_contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
             '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    commands=dict(bridge=['/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge',*bridges],
                  judge=['/usr/bin/python3',str(root/'judge_node.py'),str(out)],
                  controller=['/usr/bin/python3',str(root/'mission_node.py'),str(out)],
                  server=['gz','sim','-s','-r','--headless-rendering','-v','3',str(out/'cell.sdf')])
    children={};handles=[];started=time.monotonic();reason='wall_guard';finished=None
    try:
        for name,cmd in commands.items():
            stdout=(out/f'{name}.stdout').open('w');stderr=(out/f'{name}.stderr').open('w');handles.extend([stdout,stderr])
            children[name]=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            if name=='controller':time.sleep(1)
        while time.monotonic()-started<a.wall_limit:
            if any(proc.poll() is not None for proc in children.values()):reason='child_exit';break
            if (out/'controller_done.json').exists():
                if finished is None:finished=time.monotonic()
                if time.monotonic()-finished>3:reason='controller_completed';break
            time.sleep(.2)
    finally:
        for proc in children.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGINT)
        for proc in children.values():
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        for f in handles:f.close()
    record=dict(stop_reason=reason,wall_s=time.monotonic()-started,commands=commands,
                exits={k:v.returncode for k,v in children.items()},partition=env['GZ_PARTITION'],
                all_owned_processes_reaped=all(p.poll() is not None for p in children.values()),
                children_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    subprocess.run(['/usr/bin/python3',str(root/'score_mission.py'),str(out)],check=True)


if __name__=='__main__':main()
