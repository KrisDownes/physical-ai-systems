"""ROS2 controller. Subscribes only scans, wheel odometry, clock (never truth)."""
import collections
import csv
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from frames import compose, inverse


def ts(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def wrap(a):return math.atan2(math.sin(a),math.cos(a))
def yaw(q):return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


class Mission(Node):
    def __init__(self,out):
        super().__init__('site_mission')
        self.cfg=json.loads((out/'controller_config.json').read_text());self.out=out
        self.map_from_mission=np.array(self.cfg.get('map_from_mission',[0,0,0]))
        self.route=self.cfg.get('route',[dict(pose=g,inspection_index=i) for i,g in enumerate(self.cfg['goals'])])
        self.targets=compose(self.map_from_mission,np.array([r['pose'] for r in self.route]))
        self.route_step=0
        self.initial=compose(self.map_from_mission,np.array(self.cfg['start']))
        self.localizer=self.cfg.get('localizer','amcl')
        self.slam_pose=None;self.slam_stamp=0;self.slam_wall=0;self.slam_odom=None;self.last_odom_wall=0;self.first_odom=None
        self.tf_pose=None;self.tf_stamp=-1.;self.tf_wall=0.
        self.pose=None;self.estimate_time=None;self.last_est_wall=0;self.sim_time=0
        self.goal=0;self.dwell=None;self.odom=collections.deque(maxlen=100);self.pending=collections.deque()
        self.front=float('inf');self.last_scan_wall=0;self.done=False
        self.cmd=self.create_publisher(Twist,'/site/cmd_vel',10)
        self.event=self.create_publisher(String,'/site/events',10)
        self.create_subscription(Clock,'/clock',self.clock,10)
        self.create_subscription(Odometry,'/site/odom',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(LaserScan,'/site/scan',self.on_scan,qos_profile_sensor_data)
        if self.localizer=='slam_toolbox':
            self.create_subscription(PoseWithCovarianceStamped,'/pose',self.on_slam,10)
            self.create_subscription(PoseStamped,'/site/slam_tf_pose',self.on_tf_pose,10)
        self.timer=self.create_timer(.05,self.tick)
        root=Path(__file__).resolve().parents[1]
        command=[str(root/'vendor/offline_amcl'),str(out/'map.txt'),
                 '/dev/stdin','/dev/stdout',*map(str,self.initial),str(self.cfg.get('seed',0)),self.cfg.get('sensor_model','likelihood_field'),
                 '500','3000','.05','.1','60','.1','5']
        self.err=(out/'amcl.stderr').open('w')
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,
                                   text=True,bufsize=1)
        self.proc.stdin.write('0 0 0 8\n');self.proc.stdin.flush()
        self.estimates=queue.Queue()
        self.raw=(out/'estimates.csv').open('w')
        def reader():
            for line in self.proc.stdout:
                self.raw.write(line);self.raw.flush()
                if not line.startswith('timestamp'):self.estimates.put([float(x) for x in line.split(',')])
        self.thread=threading.Thread(target=reader,daemon=True);self.thread.start()
        self.log=(out/'controller.jsonl').open('w')

    def clock(self,m):self.sim_time=m.clock.sec+m.clock.nanosec*1e-9
    def on_odom(self,m):
        p=m.pose.pose;self.odom.append((ts(m.header),np.array([p.position.x,p.position.y,yaw(p.orientation)])))
        self.last_odom_wall=time.monotonic()
        if self.first_odom is None:self.first_odom=self.odom[-1][1].copy()
        if self.localizer=='slam_toolbox' and self.slam_pose is not None:self.associate_slam_odometry()
        self.process()
    def on_slam(self,m):
        p=m.pose.pose;self.slam_pose=np.array([p.position.x,p.position.y,yaw(p.orientation)])
        self.slam_stamp=ts(m.header);self.slam_wall=time.monotonic()
        self.slam_odom=None
        self.associate_slam_odometry()
    def on_tf_pose(self,m):
        stamp=ts(m.header)
        if stamp<=self.tf_stamp:return
        p=m.pose;self.tf_pose=np.array([p.position.x,p.position.y,yaw(p.orientation)])
        self.tf_stamp=stamp;self.tf_wall=time.monotonic()
    def associate_slam_odometry(self):
        times=np.array([row[0] for row in self.odom])
        if len(times)>=2 and times[0]<=self.slam_stamp<=times[-1]:
            hi=min(int(np.searchsorted(times,self.slam_stamp,side='right')),len(times)-1);lo=max(0,hi-1)
            if times[hi]-times[lo]<=.15:
                a,b=self.odom[lo][1],self.odom[hi][1];d=b-a;d[2]=wrap(d[2])
                self.slam_odom=a+(self.slam_stamp-times[lo])/max(1e-12,times[hi]-times[lo])*d
    def on_scan(self,m):
        self.last_scan_wall=time.monotonic()
        angles=m.angle_min+np.arange(len(m.ranges))*m.angle_increment
        ranges=np.array(m.ranges)
        front=ranges[abs(angles)<.35];self.front=float(np.min(front)) if len(front) else 0
        self.pending.append(m);self.process()
    def process(self):
        while self.pending and len(self.odom)>1:
            m=self.pending[0];t=ts(m.header)
            if t>self.odom[-1][0]:break
            self.pending.popleft()
            if t<self.odom[0][0]:continue
            times=np.array([v[0] for v in self.odom]);hi=int(np.searchsorted(times,t,side='right'))
            hi=min(hi,len(times)-1);lo=max(0,hi-1)
            if times[hi]-times[lo]>.15:continue
            a,b=self.odom[lo][1],self.odom[hi][1]
            f=(t-times[lo])/max(1e-12,times[hi]-times[lo]);delta=b-a;delta[2]=wrap(delta[2]);p=a+f*delta
            ranges=np.array(m.ranges);ranges[(~np.isfinite(ranges))|(ranges<m.range_min)|(ranges>=8)]=8
            angles=m.angle_min+np.arange(len(ranges))*m.angle_increment
            row=[t,*p,len(ranges),*np.ravel(np.c_[ranges,angles])]
            self.proc.stdin.write(' '.join(map(str,row))+'\n');self.proc.stdin.flush()

    def tick(self):
        while not self.estimates.empty():
            row=self.estimates.get();self.estimate_time=row[0];self.pose=np.array(row[1:4]);self.last_est_wall=time.monotonic()
        if self.localizer=='survey_odometry' and self.first_odom is not None:
            # Scripted survey stand-in, not measured human labor. Known dock
            # plus wheel odometry only; independent mapper receives same lidar.
            self.pose=compose(compose(self.initial,inverse(self.first_odom)),self.odom[-1][1])
            self.estimate_time=self.odom[-1][0];self.last_est_wall=self.last_odom_wall
        elif self.localizer=='slam_toolbox' and self.cfg.get('slam_pose_source','pose')=='tf':
            # Official map->odom composed at the exact current wheel timestamp
            # by slam_bridge; no second pose/odom interpolation or truth input.
            # A map->odom identity may exist before the first accepted scan.
            # Remain docked until the official mapper initializes its gauge.
            self.pose=self.tf_pose if self.slam_pose is not None else None
            self.estimate_time=self.tf_stamp;self.last_est_wall=self.tf_wall
        elif self.localizer=='slam_toolbox':
            self.pose=self.slam_pose;self.estimate_time=self.slam_stamp;self.last_est_wall=self.slam_wall
            if self.slam_pose is not None and self.slam_odom is not None:
                # Official mapper publishes /pose only on accepted motion scans.
                # Propagate its last pose with measured wheel increments so a
                # stationary dwell does not require new mapping keyframes.
                delta=compose(inverse(self.slam_odom),self.odom[-1][1])
                self.pose=compose(self.slam_pose,delta);self.estimate_time=self.odom[-1][0]
                if np.linalg.norm(delta[:2])<.3 and abs(delta[2])<.3:
                    self.last_est_wall=self.last_odom_wall
        cmd=Twist();status='waiting_for_sensors'
        fresh=self.pose is not None and time.monotonic()-self.last_est_wall<1 and time.monotonic()-self.last_scan_wall<1
        if self.localizer=='slam_toolbox' and self.cfg.get('slam_pose_source','pose')=='tf':
            fresh=fresh and -.1<=self.sim_time-self.estimate_time<1
        if fresh and not self.done:
            target=self.targets[self.route_step];d=target[:2]-self.pose[:2];distance=np.linalg.norm(d)
            if distance>self.cfg.get('arrival_tolerance_m',.09):
                angle=wrap(math.atan2(d[1],d[0])-self.pose[2]);cmd.angular.z=float(np.clip(2*angle,-.8,.8))
                if abs(angle)<.4 and self.front>.4:cmd.linear.x=float(min(.25,.8*distance))
                self.dwell=None;status='driving' if self.front>.4 else 'obstacle_stop'
            else:
                angle=wrap(target[2]-self.pose[2]);status='aligning'
                if abs(angle)>.06:cmd.angular.z=float(np.clip(2*angle,-.6,.6));self.dwell=None
                else:
                    status='dwelling'
                    # A pose tolerance alone does not establish stationary dwell.
                    stationary=False
                    if len(self.odom)>=2:
                        dt=self.odom[-1][0]-self.odom[-2][0]
                        delta=compose(inverse(self.odom[-2][1]),self.odom[-1][1])
                        stationary=(dt>0 and np.linalg.norm(delta[:2])/dt<.02 and abs(delta[2])/dt<.04)
                    if not stationary:self.dwell=None
                    if stationary and self.dwell is None:self.dwell=self.sim_time
                    inspection=self.route[self.route_step]['inspection_index']
                    dwell_s=self.cfg['dwell_s'] if inspection is not None else 0
                    if self.route_step==len(self.route)-1 and self.cfg.get('dock_completion'):
                        dwell_s=self.cfg['dock_completion']['stationary_dwell_s']
                    if dwell_s:dwell_s+=self.cfg.get('control_dwell_margin_s',.2)
                    if stationary and self.dwell is not None and self.sim_time-self.dwell>=dwell_s:
                        event=dict(type='claimed_stop' if inspection is not None else 'transit_reached',
                                   goal=inspection,route_step=self.route_step,time=self.sim_time,estimate=self.pose.tolist())
                        self.event.publish(String(data=json.dumps(event)))
                        self.route_step+=1;self.dwell=None
                        if inspection is not None:self.goal+=1
                        if self.route_step==len(self.route) or ('stop_after_goals' in self.cfg and self.goal>=self.cfg['stop_after_goals']):
                            self.done=True;self.event.publish(String(data=json.dumps(dict(type='controller_complete',time=self.sim_time))))
                            (self.out/'controller_done.json').write_text(json.dumps(dict(time=self.sim_time)))
        self.cmd.publish(cmd)
        self.log.write(json.dumps(dict(time=self.sim_time,pose=None if self.pose is None else self.pose.tolist(),
                       estimate_time=self.estimate_time,goal=self.goal,status=status,
                       slam_pose_source=self.cfg.get('slam_pose_source','pose'),
                       wheel_time=self.odom[-1][0] if self.odom else None,
                       slam_anchor_available=self.slam_odom is not None,
                       route_step=self.route_step,
                       v=cmd.linear.x,w=cmd.angular.z,front=self.front))+'\n');self.log.flush()

    def close(self):
        if rclpy.ok():self.cmd.publish(Twist())
        try:self.proc.stdin.close()
        except BrokenPipeError:pass
        try:self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:self.proc.terminate();self.proc.wait(timeout=3)
        self.thread.join(timeout=2);self.raw.close();self.err.close();self.log.close()


def main():
    rclpy.init();node=Mission(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
