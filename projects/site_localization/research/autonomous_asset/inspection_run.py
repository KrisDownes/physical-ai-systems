import json,os,shutil,subprocess,sys,time,uuid,signal,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'research/moved_asset'
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def run_one(job,freeze):
    out=Path(job['output']);shutil.copytree(job['prepared'],out)
    env=os.environ.copy();env.update(GZ_PARTITION='moved_asset_'+uuid.uuid4().hex,ROS_DOMAIN_ID=str(freeze['domain']),
        ROS_LOG_DIR=str(out/'ros_logs'),GZ_LOG_PATH=str(out/'gz_logs'),LIBGL_ALWAYS_SOFTWARE='1',
        MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',EGL_PLATFORM='surfaceless')
    bridges=['/site/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        '/site/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry','/site/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        '/judge/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry','/site/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
        '/judge/wheel_contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts','/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    commands=dict(bridge=['/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge',*bridges],
        judge=['/usr/bin/python3',str(ROOT/'sim/judge_node.py'),str(out)],
        camera=['/usr/bin/python3',str(HERE/'camera.py'),str(out)],
        controller=['/usr/bin/python3',str(ROOT/'research/autonomous_asset/inspection_policy.py'),str(out)],
        server=['gz','sim','-s','-r','--headless-rendering','-v','3',str(out/'private/world.sdf')])
    children={};handles=[];start=time.monotonic();reason='wall_guard';finished=None
    try:
        for name,cmd in commands.items():
            stdout=(out/f'{name}.stdout').open('w');stderr=(out/f'{name}.stderr').open('w');handles += [stdout,stderr]
            children[name]=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            if name=='controller':time.sleep(1)
        while time.monotonic()-start<freeze['wall_guard_s']:
            if any(p.poll() is not None for p in children.values()):reason='child_exit';break
            if (out/'task_done.json').exists():
                if finished is None:finished=time.monotonic()
                if time.monotonic()-finished>2:reason='policy_finished';break
            time.sleep(.1)
    except KeyboardInterrupt:reason='operator_abort'
    finally:
        for p in children.values():
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGINT)
                except ProcessLookupError:pass
        for p in children.values():
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in handles:f.close()
    record=dict(stop_reason=reason,wall_s=time.monotonic()-start,commands=commands,
        reset_count=1,physics_started='server' in children,partition=env['GZ_PARTITION'],domain=env['ROS_DOMAIN_ID'],
        exits={name:p.returncode for name,p in children.items()},all_owned_processes_reaped=all(p.poll() is not None for p in children.values()),
        children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save(out/'run.json',record)
    reply=subprocess.run(['/usr/bin/python3',str(HERE/'score.py'),str(out)],capture_output=True,text=True)
    (out/'score.stdout').write_text(reply.stdout);(out/'score.stderr').write_text(reply.stderr)
    record['score_returncode']=reply.returncode;save(out/'run.json',record)
    return record

