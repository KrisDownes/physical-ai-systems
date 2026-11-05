"""Analytic sensor simulator, deliberately separate from deployment policy.

Ground truth drives image generation, as in a simulator. Only image path,
request ID and timestamp leave this node on the method-facing channel.
"""
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def render(texture,pose,scene):
    width,height=scene['image_size'];focal=width/2/math.tan(math.radians(scene['horizontal_fov_deg']/2))
    x,y,heading=pose
    forward=np.array([math.cos(heading),math.sin(heading)])
    right=np.array([math.sin(heading),-math.cos(heading)])
    center=np.array(scene['asset_center']);cx,cy=scene['texture_center'];scale=scene['meters_per_texture_pixel']
    def project(world):
        d=np.array(world[:2])-[x,y];depth=float(d@forward)
        if depth<=0:raise ValueError('Fixture behind camera')
        return [width/2+focal*float(d@right)/depth,
                height/2-focal*(world[2]-scene['camera_height'])/depth]
    def point(uv):
        return center+[0,-(uv[0]-cx)*scale,-(uv[1]-cy)*scale]
    corners=np.float32([[0,0],[424,0],[424,424],[0,424]])
    h=cv2.getPerspectiveTransform(corners,np.float32([project(point(p)) for p in corners]))
    image=cv2.warpPerspective(texture,h,(width,height),borderValue=110)
    blocked=np.zeros((height,width),np.uint8)
    panel=scene.get('occluder')
    if panel:
        px,py,pz=panel['center'];sx,sy,sz=panel['size']
        # Rectangular front face is closer than the gauge for this fixed scene.
        quad=np.float32([project([px-sx/2,py+dy,pz+dz]) for dy,dz in
                         [(sy/2,sz/2),(-sy/2,sz/2),(-sy/2,-sz/2),(sy/2,-sz/2)]])
        cv2.fillConvexPoly(blocked,np.rint(quad).astype(np.int32),255)
        image[blocked>0]=180
    yy,xx=np.indices(texture.shape);rr=np.hypot(xx-cx,yy-cy)
    masks={'inner':rr<=.75*scene['radius_px'],
           'scale':(rr>=.82*scene['radius_px'])&(rr<=scene['radius_px'])}
    visibility={}
    for name,mask in masks.items():
        projected=cv2.warpPerspective(mask.astype(np.uint8),h,(width,height),flags=cv2.INTER_NEAREST)
        count=int(projected.sum())
        visibility[name]=float(np.sum((projected>0)&(blocked==0))/count) if count else 0.
    r=scene['radius_px']
    face=cv2.perspectiveTransform(np.float32([[[cx-r,cy-r],[cx+r,cy-r],[cx+r,cy+r],[cx-r,cy+r]]]),h)[0]
    diameter=float(min(np.linalg.norm(face[1]-face[0]),np.linalg.norm(face[3]-face[0])))
    fits=bool(np.all(face>=[2,2]) and np.all(face<=[width-3,height-3]))
    usable=fits and diameter>=80 and min(visibility.values())>=.95
    return image,dict(pose=pose,reference_to_image=h.tolist(),face_corners=face.tolist(),
                      face_diameter_px=diameter,face_fits=fits,visibility=visibility,
                      geometrically_usable=usable,actual_asset_id=scene['actual_asset_id'])


class Camera(Node):
    def __init__(self,out):
        super().__init__('inspection_camera_simulator');self.out=out
        self.scene=json.loads((out/'sensor_scene.json').read_text())
        self.texture=cv2.imread(self.scene['texture'],cv2.IMREAD_GRAYSCALE)
        self.pose=None;self.stamp=None;self.pending=None;self.completed=set()
        (out/'camera').mkdir()
        self.log=(out/'sensor_truth.jsonl').open('w')
        self.pub=self.create_publisher(String,'/site/camera_result',10)
        self.create_subscription(Odometry,'/judge/truth',self.on_truth,qos_profile_sensor_data)
        self.create_subscription(String,'/site/camera_request',self.request,10)

    def on_truth(self,m):
        p=m.pose.pose;q=p.orientation
        self.pose=[p.position.x,p.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
        self.stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        self.capture()

    def request(self,m):self.pending=json.loads(m.data);self.capture()

    def capture(self):
        req=self.pending
        if req is None or self.pose is None or self.stamp<req['requested_sim_time']:return
        if req['id'] in self.completed:return
        image,truth=render(self.texture,self.pose,self.scene)
        filename=f'observation_{req["id"]}.png'
        cv2.imwrite(str(self.out/'camera'/filename),image)
        truth.update(id=req['id'],time=self.stamp,file=filename)
        self.log.write(json.dumps(truth)+'\n');self.log.flush()
        self.pub.publish(String(data=json.dumps(dict(id=req['id'],time=self.stamp,file=filename))))
        self.completed.add(req['id']);self.pending=None


if __name__=='__main__':
    rclpy.init();node=Camera(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.log.close();node.destroy_node();rclpy.try_shutdown()
