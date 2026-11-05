"""Freeze and run a small matched development cohort; no hidden scene input to policy."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
BUNDLE=ROOT/'results/asset_calibration_20261023/frozen_bundle'
VISION_PY=ROOT.parent/'robot_policy_eval/.venv/bin/python3'


def load(p):return json.loads(Path(p).read_text())
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_textures(folder):
    import cv2
    import numpy as np
    folder.mkdir(parents=True,exist_ok=False)
    gauge=cv2.imread(str(BUNDLE/'reference.png'),0)
    for marker_id in [7,8]:
        image=np.full((600,1000),245,np.uint8);image[90:515,436:861]=gauge
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        image[180:420,100:340]=cv2.aruco.generateImageMarker(dictionary,marker_id,240)
        cv2.imwrite(str(folder/f'asset_{marker_id}.png'),image)


def prepare(out):
    import numpy as np
    if (out/'freeze.json').exists():raise FileExistsError('Already frozen')
    out.mkdir(parents=True,exist_ok=True);make_textures(out/'textures')
    base=load(ROOT/'sim/cell.json')
    for name,digest in load(BUNDLE/'manifest.json')['files'].items():assert sha(BUNDLE/name)==digest
    jobs=[]
    for case in ['unchanged','moved','wrong_asset','absent']:
        obstacles=json.loads(json.dumps(base['obstacles']))
        if case!='unchanged':obstacles.append([0,-.6,.7,.6])
        assets=[]
        if case=='unchanged':assets=[dict(id=7,pose=[0,-1.5,0])]
        elif case=='moved':assets=[dict(id=7,pose=[1.8,1.8,0])]
        elif case=='wrong_asset':assets=[dict(id=7,pose=[1.8,1.8,0]),dict(id=8,pose=[0,-1.5,0])]
        elif case=='absent':assets=[dict(id=8,pose=[0,-1.5,0])]
        for a in assets:a['texture']=str(out/'textures'/f'asset_{a["id"]}.png')
        for mode in ['old_location','autonomous']:
            name=case+'__'+mode;folder=out/'prepared'/name
            (folder/'private').mkdir(parents=True);(folder/'public').mkdir()
            # Standard map data is an explicitly supplied updated experimental input.
            res=.05;width=140;origin=-3.5;axis=origin+np.arange(width)*res;x,y=np.meshgrid(axis,axis)
            grid=np.full((width,width),-1)
            for ox,oy,sx,sy in obstacles:grid[(abs(x-ox)<=sx/2)&(abs(y-oy)<=sy/2)]=1
            with (folder/'map.txt').open('w') as f:
                f.write(f'{width} {width} {res} {origin} {origin}\n');np.savetxt(f,grid,fmt='%d')
            ctrl=dict(map_from_mission=[0,0,0],start=base['start'],goals=[[-1.5,-1.5,0]],
                route=[dict(pose=[-1.5,-1.5,0],inspection_index=None)],localizer='amcl',seed=7201,
                dwell_s=2,control_dwell_margin_s=.2,arrival_tolerance_m=.03,slam_pose_source='tf')
            save(folder/'controller_config.json',ctrl)
            save(folder/'public/policy.json',dict(mode=mode,old_view=[-1.5,-1.5,0],old_asset_pose=[0,-1.5,0],
                search_coordinates=[-1.5,0,1.5],inflation_m=.45,max_search_positions=4,max_captures=36,
                max_path_m=18,max_sim_s=420,track_ttl_s=60,vision_python=str(VISION_PY)))
            save(folder/'public/vision.json',dict(target_id=7,marker_size_m=.16,min_marker_edge_px=16,
                marker_to_gauge_right_m=430*.22/330,reference_width_m=425*.22/330,
                image_size=[1920,1080],horizontal_fov_deg=60,bundle=str(BUNDLE)))
            scene=dict(camera=dict(image_size=[1920,1080],horizontal_fov_deg=60,height_m=.8),
                asset_height_m=.8,gauge_center_uv=[650,300],meters_per_texture_pixel=.22/330,
                assets=assets,obstacles=[[*o,1.] for o in obstacles])
            save(folder/'private/scene.json',scene)
            save(folder/'private/evaluation.json',dict(case=case,intended_id=7,reference_interval=[6,6.5],
                tolerance_MPa=.5,stationary_dwell_s=2,distance_range_m=[.9,1.3],heading_tolerance_deg=10,
                min_face_diameter_px=80,min_visibility=.95,max_sim_s=420,max_path_m=18,
                target_present=case!='absent',physical_obstacles=obstacles))
            tree=ET.parse(ROOT/'sim/cell.sdf');world=tree.getroot().find('world')
            if case!='unchanged':
                sys.path.insert(0,str(ROOT/'sim'));from generate import box
                box(world,'relocated_cabinet',0,-.6,.7,.6)
            # Collision panels are above the base; rendered content lives in sensor simulator.
            for a in assets:
                model=ET.SubElement(world,'model',name='asset_panel_'+str(a['id']))
                ET.SubElement(model,'static').text='true'
                ax,ay,az=a['pose'];ET.SubElement(model,'pose').text=f'{ax} {ay} .8 0 0 {az}'
                link=ET.SubElement(model,'link',name='panel')
                geom=ET.SubElement(ET.SubElement(link,'collision',name='panel'),'geometry')
                ET.SubElement(ET.SubElement(geom,'box'),'size').text='.01 .667 .4'
            tree.write(folder/'private/world.sdf',encoding='unicode')
            jobs.append(dict(case=case,mode=mode,name=name,prepared=str(folder),output=str(out/name)))
    for case in ['unchanged','moved','wrong_asset','absent']:
        a=out/'prepared'/f'{case}__old_location';b=out/'prepared'/f'{case}__autonomous'
        for name in ['map.txt','controller_config.json','public/vision.json','private/scene.json','private/world.sdf','private/evaluation.json']:
            assert sha(a/name)==sha(b/name),(case,name)
        aa,bb=load(a/'public/policy.json'),load(b/'public/policy.json');aa.pop('mode');bb.pop('mode');assert aa==bb
    sources=[*HERE.glob('*.py'),ROOT/'research/controller.py',ROOT/'research/frames.py',ROOT/'sim/judge_node.py',
             ROOT/'vendor/offline_amcl',ROOT/'vendor/setup_amcl.bash',ROOT/'docs/moved_asset/contract.md',ROOT/'docs/moved_asset/protocol.md']
    sources += [p for p in BUNDLE.iterdir() if p.is_file()]
    sources += [p for p in (out/'prepared').rglob('*') if p.is_file()]+list((out/'textures').glob('*.png'))
    (out/'source').mkdir()
    for p in HERE.glob('*.py'):shutil.copyfile(p,out/'source'/p.name)
    save(out/'freeze.json',dict(frozen_utc=now(),revision='moved-asset-development-v1',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),jobs=jobs,
        files={str(p):sha(p) for p in sources},domain=213,wall_guard_s=500,
        reader_manifest_sha256=sha(BUNDLE/'manifest.json'),map_update_supplied=True))
    print('Frozen eight missions:',out,sha(out/'freeze.json'))


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
        controller=['/usr/bin/python3',str(HERE/'policy.py'),str(out)],
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


def execute(out,start_job=0,limit=None):
    freeze=load(out/'freeze.json')
    for p,h in freeze['files'].items():assert sha(p)==h,p
    lock=open(f'/tmp/site_localization_domain_{freeze["domain"]}.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Read-only collision check; never terminate another session's simulator.
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:cmd=(folder/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        except (OSError,PermissionError):continue
        if 'gz sim -s' in cmd and ('ruby' in cmd or '/gz ' in cmd):
            raise RuntimeError('Another Gazebo server is active; no new mission started: '+cmd[:180])
    jobs=freeze['jobs'][start_job:None if limit is None else start_job+limit]
    with (out/'attempts.jsonl').open('a') as ledger:
        for job in jobs:
            if Path(job['output']).exists():raise FileExistsError('Preserve attempted run '+job['output'])
            ledger.write(json.dumps(dict(event='start',utc=now(),**job))+'\n');ledger.flush()
            result=run_one(job,freeze)
            ledger.write(json.dumps(dict(event='finish',utc=now(),job=job['name'],**result))+'\n');ledger.flush()
            print(job['name'],result['stop_reason'],'score',result['score_returncode'],flush=True)
            if result['stop_reason'] in ['child_exit','operator_abort'] or result['score_returncode']:
                raise RuntimeError('Preserved infrastructure/scoring failure; stop for diagnosis')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--start-job',type=int,default=0);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.execute:execute(a.out.resolve(),a.start_job,a.limit)
    else:prepare(a.out.resolve())
