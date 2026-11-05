"""Deployment-side search, pixel identity, observed-pose approach and verification.

Reads only public config, occupancy map, camera images and estimate/sensor data.
No scenario/world/private file or simulator-truth topic is accessed.
"""
import collections
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

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from controller import Mission
sys.path.insert(0,str(ROOT/'research/moved_asset'))
from planning import GridPlanner


def wrap(a):return math.atan2(math.sin(a),math.cos(a))


class SearchMission(Mission):
    def __init__(self,out):
        self.phase='boot';self.finished=False;self.pose_history=collections.deque(maxlen=10000)
        self.worker=None;self.worker_handles=[];self.capture=None;self.capture_count=0
        self.track=None;self.first_confirmation=None;self.recoveries=0
        self.locations=[];self.heading_index=0;self.path_length=0.;self.last_wheel=None
        self.pending_goal=None;self.leg_started=0.;self.last_progress=None;self.progress_time=0.
        self.purpose='old_view';self.vision_records=[]
        self.settings=json.loads((out/'public/policy.json').read_text())
        super().__init__(out)
        self.done=True
        self.trace=(out/'method.jsonl').open('w');self.history_log=(out/'method_pose_history.jsonl').open('w')
        self.planner=GridPlanner(out/'map.txt',self.settings['inflation_m'])
        self.request=self.create_publisher(String,'/site/camera_request',10)
        self.create_subscription(String,'/site/camera_result',self.on_image,10)

    def record(self,event,**values):
        row=dict(event=event,time=self.sim_time);row.update(values)
        self.trace.write(json.dumps(row)+'\n');self.trace.flush()

    def finish(self,success,reason):
        if self.finished:return
        self.finished=True;self.done=True;self.cmd.publish(Twist())
        result=dict(success_claim=success,reason=reason,time=self.sim_time,captures=self.capture_count,
            visited_search_positions=self.locations,recovery_actions=self.recoveries,
            wheel_path_m=self.path_length,target_track=self.track,
            final_observation=self.vision_records[-1] if self.vision_records else None)
        self.record('finish',**result);(self.out/'task_done.json').write_text(json.dumps(result,indent=2)+'\n')

    def navigate(self,goal,purpose):
        plan=self.planner.path(self.pose[:2],goal[:2])
        self.record('plan',purpose=purpose,start=self.pose.tolist(),goal=goal,path=plan)
        if plan is None:return False
        points=plan['points'][1:];route=[]
        for i,xy in enumerate(points):
            yaw=goal[2] if i==len(points)-1 else math.atan2(points[i+1][1]-xy[1],points[i+1][0]-xy[0])
            route.append(dict(pose=[*xy,yaw],inspection_index=None))
        self.route=route;self.targets=np.array([r['pose'] for r in route]);self.route_step=0
        self.dwell=None;self.done=False;self.cfg['dock_completion']={'stationary_dwell_s':2.}
        self.pending_goal=goal;self.purpose=purpose;self.phase='navigation';self.leg_started=self.sim_time
        self.last_progress=self.pose[:2].copy();self.progress_time=self.sim_time
        return True

    def capture_request(self,inspect=False):
        if self.capture_count>=self.settings['max_captures']:
            self.finish(False,'search_budget_exhausted');return
        if self.request.get_subscription_count()<1:return
        self.capture=dict(id=self.capture_count,inspect=inspect,request_time=self.sim_time,
                          request_wall=time.monotonic(),purpose=self.purpose)
        self.capture_count+=1;self.phase='camera_wait'
        self.record('capture_requested',**self.capture)
        self.request.publish(String(data=json.dumps(dict(id=self.capture['id'],time=self.sim_time))))

    def on_image(self,message):
        if self.phase!='camera_wait':return
        obs=json.loads(message.data)
        if obs['id']!=self.capture['id']:return
        image=(self.out/'camera'/obs['file']).resolve()
        if image.parent!=(self.out/'camera').resolve():self.finish(False,'infrastructure_invalid_image_path');return
        self.capture.update(sensor_time=obs['time'],file=obs['file'])
        output=self.out/f'vision_{obs["id"]:03d}.json'
        cmd=[self.settings['vision_python'],str(Path(__file__).with_name('vision.py')),
             '--image',str(image),'--config',str(self.out/'public/vision.json'),'--out',str(output)]
        if self.capture['inspect']:cmd.append('--inspect')
        self.worker_handles=[(self.out/f'vision_{obs["id"]:03d}.stdout').open('w'),
                             (self.out/f'vision_{obs["id"]:03d}.stderr').open('w')]
        self.worker=subprocess.Popen(cmd,stdout=self.worker_handles[0],stderr=self.worker_handles[1])
        self.capture.update(output=str(output),worker_wall=time.monotonic());self.phase='vision_wait'

    def pose_at(self,stamp):
        rows=list(self.pose_history)
        if len(rows)<2:return None
        times=np.array([r[0] for r in rows]);hi=int(np.searchsorted(times,stamp,side='right'))
        if hi==0 or hi==len(rows):return None
        lo=hi-1
        if times[hi]-times[lo]>.15:return None
        a,b=np.array(rows[lo][1:4]),np.array(rows[hi][1:4]);delta=b-a;delta[2]=wrap(delta[2])
        pose=a+(stamp-times[lo])/(times[hi]-times[lo])*delta
        return pose,dict(bracket=[float(times[lo]),float(times[hi])],amcl_variance=rows[lo][6:9])

    def interpret(self,data):
        stamp=self.capture['sensor_time'];association=self.pose_at(stamp)
        row=dict(id=self.capture['id'],sensor_time=stamp,decision_time=self.sim_time,
                 purpose=self.capture['purpose'],vision=data)
        if association is None or self.sim_time-stamp>5:
            self.record('observation_rejected',reason='timestamp_or_processing_age',**row)
            self.finish(False,'stale_or_unassociated_observation');return
        pose,assoc=association;row.update(associated_pose=pose.tolist(),association=assoc)
        target=data.get('target')
        estimate=None
        if target:
            def horizontal(v):
                x,y,z=v;c,s=math.cos(pose[2]),math.sin(pose[2])
                return np.array([c*z+s*x,s*z-c*x])
            point=pose[:2]+horizontal(target['gauge_camera']);normal=horizontal(target['normal_camera'])
            normal/=np.linalg.norm(normal)
            estimate=dict(position=point.tolist(),normal=normal.tolist(),stamp=stamp,
                uncertainty_note='pixel sensitivity and AMCL variance logged; no calibrated joint bound',
                pixel_range_sensitivity_m=target['range_pixel_sensitivity_m'])
            row['observed_asset_map']=estimate
        self.vision_records.append(row);self.record('observation',**row)
        if data['identity_decision']=='duplicate_id_uncertain':self.finish(False,'identity_uncertain');return
        if self.purpose in ['approach','recovery']:
            if estimate is None:self.finish(False,'identity_not_reacquired');return
            self.track=estimate
            p=np.array(target['gauge_camera']);distance=float(np.linalg.norm(p[[0,2]]))
            bearing=abs(math.atan2(p[0],p[2]));reading=data.get('reading',{})
            if .9<=distance<=1.3 and bearing<=math.radians(10) and reading.get('decision')=='accept':
                self.finish(True,'fresh_identity_and_inspection');return
            if self.recoveries<1:
                self.recoveries+=1;self.plan_approach(estimate,lateral=.3);return
            self.finish(False,'inspection_unreadable_or_viewpoint_inadequate');return
        if estimate is not None:
            if self.first_confirmation is None:
                self.first_confirmation=estimate;self.phase='confirm_delay';self.confirm_after=self.sim_time+.4;return
            a=self.first_confirmation;b=estimate
            discrepancy=float(np.linalg.norm(np.array(a['position'])-b['position']))
            angular=math.acos(np.clip(np.dot(a['normal'],b['normal']),-1,1))
            self.record('target_confirmation',position_disagreement_m=discrepancy,normal_disagreement_rad=angular)
            if discrepancy>.15 or angular>math.radians(15):self.finish(False,'identity_pose_inconsistent');return
            self.track=estimate;self.first_confirmation=None;self.plan_approach(estimate);return
        self.first_confirmation=None
        if self.settings['mode']=='old_location':self.finish(False,'target_not_observed_at_old_view');return
        self.next_search()

    def plan_approach(self,estimate,lateral=0.):
        p=np.array(estimate['position']);n=np.array(estimate['normal']);side=np.array([-n[1],n[0]])
        xy=p+1.1*n+lateral*side;yaw=math.atan2(p[1]-xy[1],p[0]-xy[0])
        if not self.navigate([*xy,yaw],'recovery' if lateral else 'approach'):
            self.finish(False,'target_detected_no_reachable_approach')

    def next_search(self):
        self.heading_index+=1
        if self.heading_index<8:
            goal=[*self.locations[-1],self.heading_index*math.pi/4]
        else:
            if len(self.locations)>=self.settings['max_search_positions']:
                self.finish(False,'search_budget_exhausted');return
            goal=self.planner.next_search(self.pose[:2],self.locations,self.settings['search_coordinates'])
            if goal is None:self.finish(False,'search_candidates_exhausted_no_absence_proof');return
            self.locations.append(goal[:2]);self.heading_index=0
        if not self.navigate(goal,'search'):self.finish(False,'navigation_unreachable_search_goal')

    def tick(self):
        if self.finished:self.cmd.publish(Twist());return
        last=None
        while not self.estimates.empty():
            last=self.estimates.get();self.pose_history.append(last)
            self.history_log.write(json.dumps(last)+'\n');self.history_log.flush()
        if last is not None:self.estimates.put(last)
        super().tick()
        if self.odom:
            stamp,pose=self.odom[-1]
            if self.last_wheel is not None and stamp>self.last_wheel[0]:
                self.path_length+=float(np.linalg.norm(pose[:2]-self.last_wheel[1][:2]))
            self.last_wheel=(stamp,pose.copy())
        if self.sim_time>self.settings['max_sim_s'] or self.path_length>self.settings['max_path_m']:
            self.finish(False,'search_budget_exhausted');return
        if self.phase=='boot' and self.pose is not None and self.sim_time>=1.:
            goal=self.settings['old_view'];self.locations=[goal[:2]]
            if not self.navigate(goal,'old_view'):self.finish(False,'navigation_unreachable_old_view')
        elif self.phase=='navigation':
            if self.track and self.sim_time-self.track['stamp']>self.settings['track_ttl_s']:
                self.finish(False,'stale_target_observation');return
            if self.done:
                self.record('navigation_arrived',purpose=self.purpose,estimated_pose=self.pose.tolist(),commanded_goal=self.pending_goal)
                self.phase='request_capture'
            elif self.sim_time-self.leg_started>90:self.finish(False,'navigation_timeout')
            elif self.front<=.4:
                if self.sim_time-self.progress_time>12:self.finish(False,'navigation_obstacle_stop')
            else:self.progress_time=self.sim_time
        if self.phase=='request_capture':self.capture_request(self.purpose in ['approach','recovery'])
        elif self.phase=='confirm_delay' and self.sim_time>=self.confirm_after:self.capture_request(False)
        elif self.phase=='camera_wait' and time.monotonic()-self.capture['request_wall']>10:self.finish(False,'infrastructure_camera_timeout')
        elif self.phase=='vision_wait':
            if time.monotonic()-self.capture['worker_wall']>10:
                self.worker.kill();self.worker.wait();self.finish(False,'infrastructure_vision_timeout')
            elif self.worker.poll() is not None:
                for f in self.worker_handles:f.close()
                if self.worker.returncode:self.finish(False,'infrastructure_vision_error')
                else:self.interpret(json.loads(Path(self.capture['output']).read_text()))
                self.worker=None

    def close(self):
        if self.worker is not None and self.worker.poll() is None:self.worker.terminate();self.worker.wait(timeout=5)
        for f in self.worker_handles:
            if not f.closed:f.close()
        super().close();self.trace.close();self.history_log.close()


if __name__=='__main__':
    rclpy.init();node=SearchMission(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()
