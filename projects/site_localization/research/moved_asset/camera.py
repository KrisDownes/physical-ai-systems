"""Simulator-only state produces images; policy receives no metadata about assets."""
import json
import math
from pathlib import Path
import sys
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from render import render


class Camera(Node):
    def __init__(self,out):
        super().__init__('moved_asset_camera');self.out=out
        self.scene=json.loads((out/'private/scene.json').read_text())
        self.pose=None;self.stamp=None;self.pending=None;self.completed=set()
        (out/'camera').mkdir();self.log=(out/'private/sensor_truth.jsonl').open('w')
        self.pub=self.create_publisher(String,'/site/camera_result',10)
        self.create_subscription(Odometry,'/judge/truth',self.truth,qos_profile_sensor_data)
        self.create_subscription(String,'/site/camera_request',self.request,10)

    def truth(self,m):
        p=m.pose.pose;q=p.orientation
        self.pose=[p.position.x,p.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
        self.stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9;self.capture()

    def request(self,m):self.pending=json.loads(m.data);self.capture()

    def capture(self):
        req=self.pending
        if req is None or self.pose is None or self.stamp<req['time'] or req['id'] in self.completed:return
        image,evaluation=render(self.pose,self.scene);name=f'observation_{req["id"]:03d}.png'
        cv2.imwrite(str(self.out/'camera'/name),image)
        evaluation.update(id=req['id'],time=self.stamp,file=name)
        self.log.write(json.dumps(evaluation)+'\n');self.log.flush()
        self.pub.publish(String(data=json.dumps(dict(id=req['id'],time=self.stamp,file=name))))
        self.completed.add(req['id']);self.pending=None


if __name__=='__main__':
    rclpy.init();node=Camera(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.log.close();node.destroy_node();rclpy.try_shutdown()
