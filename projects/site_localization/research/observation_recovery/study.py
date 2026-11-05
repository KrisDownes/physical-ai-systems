"""Prepare/freeze, then execute four sequential bounded development missions."""
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

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/changed_selection_20261023/maintenance_bays__rearrangement__capped__deployment'
BUNDLE=ROOT/'results/asset_calibration_20261023/frozen_bundle'


def load(path):return json.loads(Path(path).read_text())
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(out):
    if (out/'freeze.json').exists():raise FileExistsError('Already frozen')
    sys.path.insert(0,str(ROOT/'research'))
    from frames import compose
    original=load(BASE/'controller_config.json')
    physical=load(BASE/'cell.json')
    assert original['seed']==1002
    assert sha(BUNDLE/'manifest.json')=='9baea88ebe09e0a733b92969c6dd836c2938ee30f2c96ac339f6f11e995be16c'
    for name,digest in load(BUNDLE/'manifest.json')['files'].items():assert sha(BUNDLE/name)==digest
    jobs=[]
    for condition in ['clear','obstructed']:
        for mode in ['arrival_only','recover_on_abstention']:
            name=condition+'__'+mode;prep=out/'prepared'/name;prep.mkdir(parents=True)
            cfg=json.loads(json.dumps(physical));cfg.pop('dock_completion',None)
            cfg.update(goals=[physical['goals'][0]],route=[original['route'][0]],max_sim_s=90,
                       return_loop=False,inspection_zones=['unchanged'],scenario='observation_recovery_'+condition,
                       split='observation_recovery_development',reset_count=1)
            for stale in ['route_length_m','budget','geometry_audit']:cfg.pop(stale,None)
            ctrl=json.loads(json.dumps(original));ctrl.pop('dock_completion',None)
            ctrl.update(goals=cfg['goals'],route=cfg['route'])
            save(prep/'cell.json',cfg);save(prep/'controller_config.json',ctrl)
            shutil.copyfile(BASE/'map.txt',prep/'map.txt')
            asset=compose(original['map_from_mission'],[-3.5,0,0]).tolist()
            policy=dict(mode=mode,asset_position_map=asset[:2],nominal_heading_map=asset[2],
                lateral_m=.45,map_inflation_m=.37,recovery_time_s=45,max_recovery_path_m=.75,max_sim_s=90,
                verifier_python=str(ROOT.parents[0]/'robot_policy_eval/.venv/bin/python3'),
                verifier_script=str(BUNDLE/'commissioned_gauge.py'),bundle=str(BUNDLE),
                commissioning='known dock, frozen map transform, approximate asset position and nominal view direction; no alternate view supplied')
            assert Path(policy['verifier_python']).exists()
            save(prep/'policy_config.json',policy)
            panel=dict(center=[-4.25,0,1],size=[.03,.26,.6]) if condition=='obstructed' else None
            scene=dict(asset_center=[-3.5,0,1],actual_asset_id='simulated_commissioned_R02_fixture',
                texture=str(BUNDLE/'reference.png'),texture_center=[214,210],radius_px=165,
                meters_per_texture_pixel=.22/330,image_size=[1920,1080],horizontal_fov_deg=60,
                camera_height=1.,occluder=panel)
            save(prep/'sensor_scene.json',scene)
            # This file is evaluation-only, deliberately not in policy config.
            save(prep/'evaluation_contract.json',dict(reference_interval=[6,6.5],tolerance_MPa=.5,
                intended_asset_id=scene['actual_asset_id'],note='Static texture transcription interval, not measured pressure'))
            tree=ET.parse(BASE/'cell.sdf')
            if panel:
                world=tree.getroot().find('world');model=ET.SubElement(world,'model',name='overhead_view_obstruction')
                ET.SubElement(model,'static').text='true';ET.SubElement(model,'pose').text='-4.25 0 1 0 0 0'
                link=ET.SubElement(model,'link',name='panel')
                for kind in ['collision','visual']:
                    part=ET.SubElement(link,kind,name='panel_'+kind)
                    geom=ET.SubElement(part,'geometry');box=ET.SubElement(geom,'box')
                    ET.SubElement(box,'size').text='.03 .26 .6'
            tree.write(prep/'cell.sdf',encoding='unicode')
            jobs.append(dict(name=name,condition=condition,mode=mode,prepared=str(prep),output=str(out/name)))
    for condition in ['clear','obstructed']:
        a=out/'prepared'/f'{condition}__arrival_only';b=out/'prepared'/f'{condition}__recover_on_abstention'
        for name in ['controller_config.json','cell.json','cell.sdf','map.txt','sensor_scene.json','evaluation_contract.json']:
            assert sha(a/name)==sha(b/name),(condition,name)
        pa,pb=load(a/'policy_config.json'),load(b/'policy_config.json');pa.pop('mode');pb.pop('mode');assert pa==pb
    sources=[*HERE.glob('*.py'),ROOT/'research/controller.py',ROOT/'research/frames.py',
             ROOT/'sim/judge_node.py',ROOT/'sim/score_mission.py',ROOT/'research/score.py',
             ROOT/'vendor/offline_amcl',ROOT/'vendor/setup_amcl.bash',
             ROOT/'docs/task_verification/observation_recovery_protocol.md']
    sources += [p for p in BUNDLE.iterdir() if p.is_file()]
    sources += [p for p in (out/'prepared').rglob('*') if p.is_file()]
    snap=out/'source';snap.mkdir()
    for p in sources:
        if p.parent==HERE:shutil.copyfile(p,snap/p.name)
    freeze=dict(frozen_utc=now(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        jobs=jobs,files={str(p):sha(p) for p in sources},domain=211,wall_guard_s=150,
        information_equivalence='Within condition, mode is the only differing method input; extra capture/action cost reported.',
        expected_physics_attempts=4,automatic_retries=False)
    save(out/'freeze.json',freeze);print('Frozen',len(jobs),'cases',sha(out/'freeze.json'))


def run_one(job,out,freeze):
    path=Path(job['output']);shutil.copytree(job['prepared'],path)
    env=os.environ.copy();env.update(GZ_PARTITION='site_recovery_'+uuid.uuid4().hex,
        ROS_DOMAIN_ID=str(freeze['domain']),ROS_LOG_DIR=str(path/'ros_logs'),GZ_LOG_PATH=str(path/'gz_logs'),
        LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',EGL_PLATFORM='surfaceless')
    bridges=['/site/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
             '/site/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
             '/site/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
             '/judge/truth@nav_msgs/msg/Odometry[gz.msgs.Odometry',
             '/site/contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
             '/judge/wheel_contacts@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts',
             '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock']
    commands=dict(bridge=['/opt/ros/jazzy/lib/ros_gz_bridge/parameter_bridge',*bridges],
        judge=['/usr/bin/python3',str(ROOT/'sim/judge_node.py'),str(path)],
        camera=['/usr/bin/python3',str(HERE/'camera.py'),str(path)],
        controller=['/usr/bin/python3',str(HERE/'policy.py'),str(path)],
        server=['gz','sim','-s','-r','--headless-rendering','-v','3',str(path/'cell.sdf')])
    children={};handles=[];started=time.monotonic();reason='wall_guard';finished=None
    try:
        for name,cmd in commands.items():
            stdout=(path/f'{name}.stdout').open('w');stderr=(path/f'{name}.stderr').open('w');handles.extend([stdout,stderr])
            children[name]=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            if name=='controller':time.sleep(1)
        while time.monotonic()-started<freeze['wall_guard_s']:
            if any(p.poll() is not None for p in children.values()):reason='child_exit';break
            if (path/'integration_done.json').exists():
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
    record=dict(stop_reason=reason,commands=commands,wall_s=time.monotonic()-started,
        physics_started='server' in children,reset_count=1,partition=env['GZ_PARTITION'],domain=env['ROS_DOMAIN_ID'],
        exits={k:p.returncode for k,p in children.items()},all_owned_processes_reaped=all(p.poll() is not None for p in children.values()),
        children_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    save(path/'run.json',record)
    score=subprocess.run(['/usr/bin/python3',str(HERE/'score.py'),str(path)],capture_output=True,text=True)
    (path/'score.stdout').write_text(score.stdout);(path/'score.stderr').write_text(score.stderr)
    record['score_returncode']=score.returncode;save(path/'run.json',record)
    return record


def execute(out):
    freeze=load(out/'freeze.json')
    for p,h in freeze['files'].items():assert sha(p)==h,p
    lock=open(f'/tmp/site_localization_domain_{freeze["domain"]}.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (out/'attempts.jsonl').open('x') as ledger:
        for job in freeze['jobs']:
            ledger.write(json.dumps(dict(event='start',time=now(),**job))+'\n');ledger.flush()
            result=run_one(job,out,freeze)
            ledger.write(json.dumps(dict(event='finished',time=now(),job=job['name'],**result))+'\n');ledger.flush()
            print(job['name'],result['stop_reason'],'score',result['score_returncode'],flush=True)
            if result['stop_reason'] in ['child_exit','operator_abort'] or result['score_returncode']:
                raise RuntimeError('Attempt retained; infrastructure/implementation failure stops collection')
    save(out/'collection_complete.json',dict(attempts=4,resets=4,retries=0,finished_utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    (execute if a.execute else prepare)(a.out.resolve())
