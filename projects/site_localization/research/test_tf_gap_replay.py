"""Regression replay of the actual development-A timestamp-gap deadlock.

No ROS graph/physics/native AMCL is started. Feed recorded method observations
into the controller's real synchronization and control methods. This checks
liveness, not physical mission effectiveness.
"""
import collections
import csv
import io
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
import numpy as np
from controller import Mission
from geometry_msgs.msg import PoseStamped

ROOT=Path(__file__).resolve().parents[1]
TRACE=ROOT/'results/correction_development_20261022/maintenance_bays__unchanged__repeat1__acquisition'


class Publisher:
    def publish(self,message):pass


class GapReplay(unittest.TestCase):
    def test_exact_tf_completes_dwell_where_missing_pose_anchor_stalls(self):
        def read(name):
            with (TRACE/name).open() as stream:return list(csv.DictReader(stream))
        wheels=read('mapper/wheel_odometry.csv')
        tf=read('mapper/mapper_tf_pose.csv')
        mapper=read('mapper/mapper_pose.csv')[-1]
        prefix=[r for r in wheels if 268.3<=float(r['timestamp'])<=270.018]
        timestamps=[float(r['timestamp']) for r in prefix]
        lo=max(t for t in timestamps if t<=268.8);hi=min(t for t in timestamps if t>=268.8)
        self.assertAlmostEqual(lo,268.430);self.assertAlmostEqual(hi,268.838)
        self.assertGreater(hi-lo,.15)
        with tempfile.TemporaryDirectory() as directory:
            def instance(source):
                m=Mission.__new__(Mission)
                m.cfg={'slam_pose_source':source,'dwell_s':2,'control_dwell_margin_s':.2,
                    'arrival_tolerance_m':.03,'dock_completion':{'stationary_dwell_s':2}}
                m.localizer='slam_toolbox';m.estimates=queue.Queue();m.slam_odom=None
                m.slam_pose=np.array([float(mapper[k]) for k in ['x','y','yaw']]);m.slam_stamp=268.8
                m.slam_wall=time.monotonic()-2;m.last_odom_wall=time.monotonic()
                m.odom=collections.deque((float(r['timestamp']),np.array([float(r[k]) for k in ['x','y','yaw']])) for r in prefix)
                m.pose=None;m.estimate_time=None;m.last_est_wall=0;m.sim_time=270.018
                m.last_scan_wall=time.monotonic();m.done=False;m.front=3.
                m.route=[{'inspection_index':None} for _ in range(13)];m.targets=np.zeros((13,3));m.route_step=12
                m.goal=6;m.dwell=None;m.cmd=Publisher();m.event=Publisher();m.log=io.StringIO();m.out=Path(directory)
                m.tf_pose=None;m.tf_stamp=-1;m.tf_wall=0
                m.associate_slam_odometry();self.assertIsNone(m.slam_odom)
                return m
            old=instance('pose');old.tick()
            self.assertEqual(json.loads(old.log.getvalue().splitlines()[-1])['status'],'waiting_for_sensors')
            corrected=instance('tf')
            for stamp in [270.018,272.4]:
                corrected.odom=collections.deque(((float(r['timestamp']),np.array([float(r[k]) for k in ['x','y','yaw']]))
                    for r in wheels if float(r['timestamp'])<=stamp),maxlen=100)
                row=max((r for r in tf if float(r['timestamp'])<=stamp),key=lambda r:float(r['timestamp']))
                message=PoseStamped();ts=float(row['timestamp']);message.header.stamp.sec=int(ts)
                message.header.stamp.nanosec=round((ts-int(ts))*1e9)
                message.pose.position.x=float(row['x']);message.pose.position.y=float(row['y'])
                message.pose.orientation.z=np.sin(float(row['yaw'])/2);message.pose.orientation.w=np.cos(float(row['yaw'])/2)
                corrected.sim_time=stamp;corrected.on_tf_pose(message);corrected.tick()
            self.assertTrue(corrected.done)
            self.assertTrue((Path(directory)/'controller_done.json').exists())
            self.assertEqual(corrected.goal,6)


if __name__=='__main__':unittest.main()
