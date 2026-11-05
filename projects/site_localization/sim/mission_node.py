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


def ts(h):return h.stamp.sec+h.stamp.nanosec*1e-9
def wrap(a):return math.atan2(math.sin(a),math.cos(a))
def yaw(q):return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


class Mission(Node):
    def __init__(self,out):
        super().__init__('site_mission')
        self.cfg=json.loads((out/'cell.json').read_text());self.out=out
        self.pose=None;self.estimate_time=None;self.last_est_wall=0;self.sim_time=0
        self.goal=0;self.dwell=None;self.odom=collections.deque(maxlen=100);self.pending=collections.deque()
        self.front=float('inf');self.last_scan_wall=0;self.done=False
        self.cmd=self.create_publisher(Twist,'/site/cmd_vel',10)
        self.event=self.create_publisher(String,'/site/events',10)
        self.create_subscription(Clock,'/clock',self.clock,10)
        self.create_subscription(Odometry,'/site/odom',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(LaserScan,'/site/scan',self.on_scan,qos_profile_sensor_data)
        self.timer=self.create_timer(.05,self.tick)
        root=Path(__file__).resolve().parents[1]
        command=[str(root/'vendor/offline_amcl'),str(out/'map.txt'),
                 '/dev/stdin','/dev/stdout',*map(str,self.cfg['start']),'0','likelihood_field',
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
        self.process()
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
        cmd=Twist();status='waiting_for_sensors'
        fresh=self.pose is not None and time.monotonic()-self.last_est_wall<1 and time.monotonic()-self.last_scan_wall<1
        if fresh and not self.done:
            target=np.array(self.cfg['goals'][self.goal]);d=target[:2]-self.pose[:2];distance=np.linalg.norm(d)
            if distance>.09:
                angle=wrap(math.atan2(d[1],d[0])-self.pose[2]);cmd.angular.z=float(np.clip(2*angle,-.8,.8))
                if abs(angle)<.4 and self.front>.4:cmd.linear.x=float(min(.25,.8*distance))
                self.dwell=None;status='driving' if self.front>.4 else 'obstacle_stop'
            else:
                angle=wrap(target[2]-self.pose[2]);status='aligning'
                if abs(angle)>.06:cmd.angular.z=float(np.clip(2*angle,-.6,.6));self.dwell=None
                else:
                    status='dwelling'
                    if self.dwell is None:self.dwell=self.sim_time
                    if self.sim_time-self.dwell>=self.cfg['dwell_s']:
                        event=dict(type='claimed_stop',goal=self.goal,time=self.sim_time,estimate=self.pose.tolist())
                        self.event.publish(String(data=json.dumps(event)))
                        self.goal+=1;self.dwell=None
                        if self.goal==len(self.cfg['goals']):
                            self.done=True;self.event.publish(String(data=json.dumps(dict(type='controller_complete',time=self.sim_time))))
                            (self.out/'controller_done.json').write_text(json.dumps(dict(time=self.sim_time)))
        self.cmd.publish(cmd)
        self.log.write(json.dumps(dict(time=self.sim_time,pose=None if self.pose is None else self.pose.tolist(),
                       estimate_time=self.estimate_time,goal=self.goal,status=status,
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
