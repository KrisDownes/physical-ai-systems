"""ROS adapter for destination selection; local planner/controller remain unchanged."""
import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as Camera
from std_msgs.msg import String
from PIL import Image, ImageDraw
from rover_exploration.frontier_node import FrontierDetector
from rover_exploration.goal_selector import GoalExecutive
from visual_rover_agent.onboard import Onboard, stamp, yaw


class HierarchicalController(FrontierDetector):
    def __init__(self):
        super().__init__()
        self.method=os.environ.get('ROVER_SELECTOR','mock')
        self.folder=Path(os.environ['ROVER_HIERARCHY_LOG'])
        self.log=(self.folder/'executive.jsonl').open('a',buffering=1)
        self.log_lock=threading.Lock(); self.requests=queue.Queue(maxsize=1)
        self.responses=queue.Queue(); self.shutdown_event=threading.Event()
        self.driver=None; self.camera=None; self.odom=None; self.guarded=None
        self.map_version=0; self.latest_map=None; self.last_sim=None
        self.last_clock_wall=time.monotonic(); self.was_safe=False; self.safe_since=None
        self.idle_since=time.monotonic(); self.idle_cause=None; self.last_path=None; self.previous_map_odom=None
        self.sensor_wall_limit=self._parameter('executive.sensor_timeout_wall_s',1.0)
        self.clock_limit=self._parameter('executive.clock_stall_wall_s',1.0)
        self.resume_debounce=self._parameter('executive.resume_debounce_wall_s',1.0)
        self.velocity_variance_limit=self._parameter('executive.velocity_variance_limit',0.5)
        self.map_jump_m=self._parameter('executive.map_jump_m',0.25)
        self.map_jump_rad=self._parameter('executive.map_jump_rad',math.radians(5))
        self.policy.obsolete_debounce_s=self._parameter('executive.obsolete_debounce_sim_s',2.0)
        self.executive=GoalExecutive(self.event,self.request,
            self._parameter('executive.request_debounce_wall_s',1.0),
            self._parameter('executive.response_timeout_wall_s',60.0))
        if self.method!='classical': self.policy.selector=self.executive
        self.onboard=Onboard(self)
        self.create_subscription(Camera,'/camera/image_raw',lambda m:setattr(self,'camera',(m,time.monotonic())),qos_profile_sensor_data)
        self.create_subscription(Odometry,'/odometry/filtered',lambda m:setattr(self,'odom',(m,time.monotonic())),10)
        self.create_subscription(Twist,'/cmd_vel_guarded',lambda m:setattr(self,'guarded',(m,time.monotonic())),10)
        self.velocity=self.create_publisher(Twist,'/cmd_vel',10)
        self.status=self.create_publisher(String,'/hierarchy_status',10)
        self.create_subscription(String,'/hierarchy_cancel',lambda m:self.cancel(m.data),10)
        self.safety_timer=self.create_timer(.05,self.safety_tick,clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.worker=threading.Thread(target=self.work,daemon=True);self.worker.start()
        self.event('configured',method=self.method,policy_config=vars(self.policy.config),
                   model='gpt-5.6-luna' if self.method=='llm' else None,effort='low',
                   inference_request_cap_enforceable=False if self.method=='llm' else None)

    def event(self,kind,**data):
        with self.log_lock:
            self.log.write(json.dumps(dict(kind=kind,wall_s=time.monotonic(),sim_s=self.node_time_s(),**{k:v for k,v in data.items() if k not in ('sim_s','wall_s')}))+'\n')
        if kind in ('cancel','safety_stop') and self.driver:
            threading.Thread(target=self.driver.cancel,daemon=True).start()

    def cancel(self,reason):
        self.executive.cancel(reason)
        if rclpy.ok(): self.velocity.publish(Twist())

    def safety_reason(self):
        now=time.monotonic();sim=self.node_time_s()
        if self.last_sim is None or sim>self.last_sim:self.last_clock_wall=now
        self.last_sim=sim
        if now-self.last_clock_wall>self.clock_limit:return 'clock_stall'
        for key,item in [('camera',self.camera),('scan',self.onboard.scan),('odom',self.odom)]:
            if item is None or now-item[1]>self.sensor_wall_limit or abs(sim-stamp(item[0]))>self.sensor_wall_limit:return key+'_stale'
        if self.latest_map is None or abs(sim-stamp(self.latest_map))>5:return 'map_stale'
        if self.latest_map.header.frame_id!='map' or abs(yaw(self.latest_map.info.origin.orientation))>1e-6:return 'unsupported_map_frame'
        from visual_rover_agent.onboard import clearances
        if any(v['state']=='missing_or_invalid' for v in clearances(self.onboard.scan[0]).values()):return 'scan_invalid'
        # EKF intentionally fuses forward velocity/yaw rate only: absolute odom
        # position covariance grows even when SLAM map localization is stable.
        cov=self.odom[0].twist.covariance
        if any(not math.isfinite(cov[i]) or cov[i]<0 or cov[i]>self.velocity_variance_limit for i in (0,35)):return 'velocity_estimate_covariance'
        try:
            tf=self.tf_buffer.lookup_transform('map','base_footprint',Time())
            correction=self.tf_buffer.lookup_transform('map','odom',Time())
            current=(correction.transform.translation.x,correction.transform.translation.y,yaw(correction.transform.rotation))
            previous=self.previous_map_odom;self.previous_map_odom=current
            if not all(math.isfinite(v) for v in current):return 'localization_invalid'
            if previous and (math.hypot(current[0]-previous[0],current[1]-previous[1])>self.map_jump_m or abs(math.atan2(math.sin(current[2]-previous[2]),math.cos(current[2]-previous[2])))>self.map_jump_rad):return 'localization_jump'
            vals=[tf.transform.translation.x,tf.transform.translation.y,yaw(tf.transform.rotation)]
            if abs(sim-stamp(tf))>.5 or not all(math.isfinite(x) for x in vals):return 'localization_stale'
        except Exception:return 'localization_missing'
        return None

    def safety_tick(self):
        reason=self.safety_reason();now=time.monotonic()
        if reason:self.safe_since=None
        elif self.safe_since is None:self.safe_since=now
        safe=not reason and now-self.safe_since>=self.resume_debounce
        self.was_safe=safe;self.executive.safety(safe)
        while not self.responses.empty():
            request_id,answer,error=self.responses.get_nowait()
            self.executive.respond(request_id,answer,error)
        if self.policy.complete and not self.executive.closed:
            self.executive.cancel('completion',complete=True)
        active=self.policy.target is not None and self.policy.target.goal_world is not None
        if safe and active and not self.executive.closed and self.executive.pending is None:self.executive.state='executing'
        cause=('complete' if self.policy.complete else 'stopped' if self.executive.closed else
               'safety:'+str(reason or 'resume_debounce') if not safe else
               'local_recovery' if self.policy.recovery_state!='idle' else
               ('waiting_for_model' if getattr(self,'method','mock')=='llm' else 'waiting_for_mock') if not active and self.executive.pending else
               'awaiting_valid_goal' if not active else 'controller_stall' if not self.guarded or now-self.guarded[1]>.3 else 'executing')
        if cause!=self.idle_cause:
            if self.idle_cause:self.event('state_duration',cause=self.idle_cause,duration_wall_s=now-self.idle_since)
            self.idle_cause=cause;self.idle_since=now;self.event('state',cause=cause)
        allowed=safe and not self.executive.closed and not self.policy.complete and (active or (self.policy.recovery_state!='idle' and self.executive.pending is None))
        # No recovery movement while a necessary destination decision is pending.
        if allowed and self.guarded and now-self.guarded[1]<=.3:
            command=self.guarded[0]
            if math.isfinite(command.linear.x) and math.isfinite(command.angular.z) and abs(command.linear.x)<=.15+1e-9 and abs(command.angular.z)<=.60+1e-9:
                self.velocity.publish(command)
            else:
                self.event('invalid_controller_command');self.velocity.publish(Twist())
        else:self.velocity.publish(Twist())
        self.status.publish(String(data=json.dumps(dict(state=self.executive.state,cause=cause,
            terminal=self.executive.closed,reason=self.executive.reason,requests=self.executive.request_count,arrivals=self.policy.counters.goals_reached))))

    def map_callback(self,msg):
        # Base constructor can receive no callbacks before executor starts.
        self.latest_map=msg;self.map_version+=1
        if not self.was_safe or self.executive.closed:
            self._publish_path(msg.header,msg.info,None);return
        self.policy.selection_context=dict(wall_s=time.monotonic(),map_version=self.map_version)
        super().map_callback(msg)

    def _publish_path(self,header,info,path):
        signature=tuple(path) if path else None
        if signature and signature!=self.last_path:self.event('path_replanned',cells=len(path))
        self.last_path=signature
        super()._publish_path(header,info,path)

    def _log_transition(self,update):
        for name in ('goal_reached','goal_assigned','failure'):
            value=getattr(update,name)
            if value is not None:
                self.event(name,value=value,goal_version=self.executive.goal_version,estimated_pose=self.latest_pose);self.executive.event(name,value=value)
        super()._log_transition(update)

    def stuck_check_callback(self):
        if not self.was_safe or self.executive.closed:return
        before=self.policy.counters.recovery_requests
        super().stuck_check_callback()
        if self.policy.counters.recovery_requests>before:
            self.executive.event('bounded_progress_failure');self.event('recovery_requested')

    def request(self,decision):
        # Capture only onboard data on the ROS thread; worker never reads project state.
        msg=self.camera[0]
        self.onboard.grid=(self.latest_map,time.monotonic())
        snapshot,map_image=self.onboard.snapshot(stamp(msg))
        channels={'rgb8':('RGB',3),'bgr8':('BGR',3),'rgba8':('RGBA',4),'bgra8':('BGRA',4)}
        if msg.encoding not in channels:
            self.executive.respond(decision['request_id'],error='camera_encoding');return
        mode,n=channels[msg.encoding]
        im=Image.frombytes('RGB' if n==3 else 'RGBA',(msg.width,msg.height),bytes(msg.data),'raw',mode,msg.step)
        buf=BytesIO();im.convert('RGB').save(buf,format='JPEG',quality=80)
        # Overlay the actual eligible approach IDs on the existing SLAM raster.
        raster=Image.open(BytesIO(base64.b64decode(map_image))).convert('RGB');draw=ImageDraw.Draw(raster)
        info=self.latest_map.info;scale=min(640/info.width,640/info.height)
        for c in decision['candidates']:
            x=(c['x_m']-info.origin.position.x)/info.resolution*scale
            y=60+round(info.height*scale)-(c['y_m']-info.origin.position.y)/info.resolution*scale
            draw.ellipse((x-4,y-4,x+4,y+4),fill='green');draw.text((x+5,y-12),c['id'],fill='green')
        mapped=BytesIO();raster.save(mapped,format='PNG');map_image=base64.b64encode(mapped.getvalue()).decode()
        snapshot['slam_map'].pop('frontiers',None)
        snapshot.update(decision,camera=dict(captured_sim_time_s=stamp(msg),frame=msg.header.frame_id,state='valid'),
            observation_id=f"goal-observation-{decision['request_id']}",
            candidate_definition='Eligible reachable approach points from shared online planner; IDs valid only for this request. Green A labels are eligible approaches; red F labels are raw frontier representatives.')
        payload=dict(snapshot=snapshot,images=['data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode(), 'data:image/png;base64,'+map_image])
        with (self.folder/'goal_observations.jsonl').open('a') as f:f.write(json.dumps(payload)+'\n')
        try:self.requests.put_nowait(payload)
        except queue.Full:self.executive.respond(decision['request_id'],error='request_queue_full')

    def work(self):
        while not self.shutdown_event.is_set():
            try:payload=self.requests.get(timeout=.1)
            except queue.Empty:continue
            rid=payload['snapshot']['request_id']
            if self.shutdown_event.is_set() or not self.executive.pending or self.executive.pending['request_id']!=rid: continue
            try:
                if self.method=='mock':
                    pose=payload['snapshot']['estimated_pose']
                    answer=min(payload['snapshot']['candidates'],key=lambda c:((c['x_m']-pose['x_m'])**2+(c['y_m']-pose['y_m'])**2,c['id']))['id']
                elif self.method=='llm':
                    from hierarchical_driver import PersistentDriver
                    if self.driver is None:self.driver=PersistentDriver(self.folder,enable_model=os.environ.get('ROVER_ENABLE_MODEL')=='1')
                    if self.shutdown_event.is_set() or not self.executive.pending or self.executive.pending['request_id']!=rid: continue
                    answer=self.driver.choose(payload)
                else:raise RuntimeError('unexpected_selector_request')
                self.event('selector_response',request_id=rid,destination_id=answer,selector=self.method)
                self.responses.put((rid,answer,None))
            except Exception as error:self.responses.put((rid,None,str(error)))

    def close(self):
        self.cancel('shutdown');self.shutdown_event.set()
        if self.driver:self.driver.close()
        self.worker.join(2)
        self.event('state_duration',cause=self.idle_cause,duration_wall_s=time.monotonic()-self.idle_since)
        self.log.close()


def main():
    rclpy.init();node=HierarchicalController()
    try:rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
