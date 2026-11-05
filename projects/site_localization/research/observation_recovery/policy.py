"""One observation recovery action. No truth, hidden scene, or reading labels.

Reuse the baseline AMCL/waypoint controller unchanged; only its mission sequence
is supervised. Camera messages contain pixels (file path) and timestamps only.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research'))
from controller import Mission
from frames import compose, inverse


def path_free(map_path, start, end, inflation):
    with Path(map_path).open() as f:
        width,height,res,ox,oy = np.fromstring(f.readline(),sep=' ')
        cells = np.loadtxt(f,dtype=int).reshape(int(height),int(width))
    # Native map samples are centered at origin + index*resolution.
    radius = inflation+math.sqrt(2)*res/2
    n = int(math.ceil(radius/res))+1
    offsets = [(x,y) for y in range(-n,n+1) for x in range(-n,n+1)]
    for p in np.linspace(start,end,max(2,int(np.ceil(np.linalg.norm(end-start)/(res/2)))+1)):
        ix,iy = np.rint((p-[ox,oy])/res).astype(int)
        for dx,dy in offsets:
            x,y=ix+dx,iy+dy
            if np.linalg.norm(np.array([ox+x*res,oy+y*res])-p)>radius:continue
            if not(0<=x<width and 0<=y<height) or cells[y,x]!=-1:return False
    return True


class Inspection(Mission):
    def __init__(self,out):
        self.phase='navigation';self.request_wall=None;self.recovery_started=None
        self.recovery_path=0.;self.last_wheel=None;self.recovery_goal=None
        self.observations=[];self.finished=False
        self.settings=json.loads((out/'policy_config.json').read_text())
        super().__init__(out)
        self.trace=(out/'integration.jsonl').open('w')
        self.camera=self.create_publisher(String,'/site/camera_request',10)
        self.create_subscription(String,'/site/camera_result',self.on_image,10)

    def record(self,kind,**data):
        row=dict(type=kind,sim_time=self.sim_time,**data)
        self.trace.write(json.dumps(row)+'\n');self.trace.flush()

    def finish(self,claim,reason):
        if self.finished:return
        self.finished=True;self.done=True;self.phase='finished';self.cmd.publish(Twist())
        result=dict(claimed_task_complete=claim,reason=reason,time=self.sim_time,
                    recovery_attempted=self.recovery_started is not None,
                    recovery_goal_map=self.recovery_goal,recovery_wheel_path_m=self.recovery_path,
                    observations=self.observations,identity_verified_by_method=False)
        self.record('finished',**result)
        (self.out/'integration_done.json').write_text(json.dumps(result,indent=2)+'\n')

    def request_image(self):
        if self.camera.get_subscription_count()<1:return
        self.request_wall=time.monotonic();self.phase='waiting_image'
        request=dict(id=len(self.observations),requested_sim_time=self.sim_time)
        self.record('camera_request',**request)
        self.camera.publish(String(data=json.dumps(request)))

    def on_image(self,message):
        if self.phase!='waiting_image':return
        msg=json.loads(message.data)
        if msg['id']!=len(self.observations):return
        image=(self.out/'camera'/msg['file']).resolve()
        if image.parent!=(self.out/'camera').resolve():
            self.finish(False,'invalid_camera_path');return
        output=self.out/f'verifier_{msg["id"]}.json'
        command=[self.settings['verifier_python'],self.settings['verifier_script'],
                 '--bundle',self.settings['bundle'],'--image',str(image),'--out',str(output)]
        start=time.monotonic()
        try:
            reply=subprocess.run(command,capture_output=True,text=True,timeout=8)
            (self.out/f'verifier_{msg["id"]}.stdout').write_text(reply.stdout)
            (self.out/f'verifier_{msg["id"]}.stderr').write_text(reply.stderr)
            if reply.returncode:raise RuntimeError('Verifier returncode '+str(reply.returncode))
            verdict=json.loads(output.read_text())
        except Exception as error:
            self.record('verifier_error',error=repr(error));self.finish(False,'verifier_error');return
        observation=dict(id=msg['id'],capture_time=msg['time'],file=msg['file'],
                         decision=verdict['decision'],reason=verdict['reason'],
                         reading_value=verdict.get('reading_value'),verifier_wall_s=time.monotonic()-start)
        self.observations.append(observation);self.record('observation',**observation)
        if self.settings['mode']=='arrival_only':self.finish(True,'arrival_only_claim');return
        if verdict['decision']=='accept':self.finish(True,'conditional_measurement_accepted');return
        if verdict['decision']!='abstain' or len(self.observations)>=2:
            self.finish(False,'operator_help_after_unresolved_observation');return
        # The next timer drains fresh AMCL estimates before choosing the action.
        self.phase='choose_recovery'

    def choose_recovery(self):
        if self.pose is None:return
        s=self.settings;anchor=np.array(s['asset_position_map'])
        yaw=s['nominal_heading_map'];left=np.array([-math.sin(yaw),math.cos(yaw)])
        candidates=[]
        for sign in [1,-1]:
            xy=self.pose[:2]+sign*s['lateral_m']*left
            target=np.array([*xy,math.atan2(anchor[1]-xy[1],anchor[0]-xy[0])])
            free=path_free(self.out/'map.txt',self.pose[:2],target[:2],s['map_inflation_m'])
            candidates.append(dict(sign=sign,target_map=target.tolist(),map_free=free))
            if not free:continue
            self.record('recovery_selected',candidates=candidates,estimate=self.pose.tolist())
            self.recovery_goal=target.tolist()
            self.recovery_started=self.sim_time;self.recovery_path=0.;self.last_wheel=None
            # Reuse original control with an explicit stationary observation dwell.
            mission_target=compose(inverse(self.map_from_mission),target).tolist()
            self.route=[dict(pose=mission_target,inspection_index=None)]
            self.targets=np.array([target]);self.route_step=0;self.done=False;self.dwell=None
            self.cfg['dock_completion']={'stationary_dwell_s':self.cfg['dwell_s']}
            self.phase='recovery_navigation';return
        self.record('recovery_unavailable',candidates=candidates)
        self.finish(False,'operator_help_no_map_free_action')

    def tick(self):
        if self.finished:
            self.cmd.publish(Twist());return
        # Always drain estimates and log through unchanged baseline implementation.
        super().tick()
        if self.phase in ['navigation','recovery_navigation'] and self.done:
            self.record('navigation_arrived',phase=self.phase,estimate=self.pose.tolist())
            self.phase='request_image';self.request_wall=time.monotonic()
        if self.phase=='request_image':self.request_image()
        elif self.phase=='choose_recovery':self.choose_recovery()
        if self.recovery_started is not None and self.odom:
            stamp,pose=self.odom[-1]
            if self.last_wheel is not None and stamp>self.last_wheel[0]:
                self.recovery_path+=float(np.linalg.norm(pose[:2]-self.last_wheel[1][:2]))
            self.last_wheel=(stamp,pose.copy())
            if self.sim_time-self.recovery_started>self.settings['recovery_time_s']:
                self.finish(False,'operator_help_recovery_timeout')
            elif self.recovery_path>self.settings['max_recovery_path_m']:
                self.finish(False,'operator_help_motion_bound')
        if self.phase in ['waiting_image','request_image'] and time.monotonic()-self.request_wall>10:
            self.finish(False,'camera_timeout')
        if self.sim_time>self.settings['max_sim_s']:self.finish(False,'mission_deadline')

    def close(self):
        super().close();self.trace.close()


if __name__=='__main__':
    rclpy.init();node=Inspection(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()
