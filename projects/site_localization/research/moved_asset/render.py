"""Analytic camera sensor from simulator geometry, not a policy input generator.

All panels are rendered on every request, with geometric occlusion. No target
selection or success information is used to choose the returned pixels.
"""
import math
from pathlib import Path
import cv2
import numpy as np


def blocked_segments(camera,points,boxes):
    direction=points-camera;blocked=np.zeros(len(points),bool)
    for box in boxes:
        ox,oy,sx,sy,height=box
        low=np.array([ox-sx/2,oy-sy/2,0.]);high=np.array([ox+sx/2,oy+sy/2,height])
        near=np.zeros(len(points));far=np.ones(len(points));valid=np.ones(len(points),bool)
        for axis in range(3):
            d=direction[:,axis];parallel=abs(d)<1e-10
            if camera[axis]<low[axis] or camera[axis]>high[axis]:valid&=~parallel
            a=np.divide(low[axis]-camera[axis],d,out=np.full(len(d),-np.inf),where=~parallel)
            b=np.divide(high[axis]-camera[axis],d,out=np.full(len(d),np.inf),where=~parallel)
            near=np.maximum(near,np.minimum(a,b));far=np.minimum(far,np.maximum(a,b))
        blocked|=valid&(near<=far)&(near<.999)&(far>1e-4)
    return blocked


def render(pose,scene):
    width,height=scene['camera']['image_size']
    f=width/2/math.tan(math.radians(scene['camera']['horizontal_fov_deg']/2))
    x,y,yaw=pose;camera=np.array([x,y,scene['camera']['height_m']])
    right=np.array([math.sin(yaw),-math.cos(yaw),0.]);forward=np.array([math.cos(yaw),math.sin(yaw),0.])
    image=np.full((height,width),110,np.uint8);depth_buffer=np.full((height,width),np.inf,np.float32)
    owner=np.full((height,width),-1,np.int16);objects=[]
    for index,asset in enumerate(scene['assets']):
        gx,gy,face_yaw=asset['pose'];center=np.array([gx,gy,scene['asset_height_m']])
        basis_right=np.array([math.sin(face_yaw),-math.cos(face_yaw),0.])
        outward=np.array([-math.cos(face_yaw),-math.sin(face_yaw),0.])
        texture=cv2.imread(asset['texture'],cv2.IMREAD_GRAYSCALE);th,tw=texture.shape
        center_uv=np.array(scene['gauge_center_uv']);scale=scene['meters_per_texture_pixel']
        def points(uv):
            uv=np.asarray(uv)
            return center+(uv[...,0,None]-center_uv[0])*scale*basis_right+(uv[...,1,None]-center_uv[1])*scale*np.array([0,0,-1.])
        corners=np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]])
        world=points(corners);delta=world-camera;depth=delta@forward
        row=dict(actual_id=asset['id'],asset_pose=asset['pose'],rendered=False,
                 distance_m=float(np.linalg.norm(center[:2]-camera[:2])),
                 face_angle_rad=float(math.acos(np.clip(np.dot(outward,(camera-center))/max(np.linalg.norm(camera-center),1e-9),-1,1))))
        objects.append(row)
        if np.min(depth)<.05 or np.dot(outward,camera-center)<=0:continue
        uv=np.c_[width/2+f*(delta@right)/depth,height/2-f*delta[:,2]/depth].astype(np.float32)
        h=cv2.getPerspectiveTransform(corners,uv)
        warped=cv2.warpPerspective(texture,h,(width,height),borderValue=110)
        mask=cv2.warpPerspective(np.ones((th,tw),np.uint8),h,(width,height),flags=cv2.INTER_NEAREST)
        ys,xs=np.nonzero(mask)
        if not len(xs):continue
        source=cv2.perspectiveTransform(np.float32(np.c_[xs,ys])[None],np.linalg.inv(h))[0]
        wp=points(source);ds=(wp-camera)@forward
        blocked=blocked_segments(camera,wp,scene['obstacles'])
        visible=~blocked&(ds<depth_buffer[ys,xs])
        image[ys[visible],xs[visible]]=warped[ys[visible],xs[visible]]
        depth_buffer[ys[visible],xs[visible]]=ds[visible];owner[ys[visible],xs[visible]]=index
        row.update(rendered=True,homography=h.tolist())
    for index,row in enumerate(objects):
        if not row['rendered']:
            row['usable_image']=False;continue
        h=np.array(row['homography']);cx,cy=scene['gauge_center_uv'];r=165
        face=cv2.perspectiveTransform(np.float32([[[cx-r,cy-r],[cx+r,cy-r],[cx+r,cy+r],[cx-r,cy+r]]]),h)[0]
        fits=bool(np.all(face>=[2,2]) and np.all(face<=[width-3,height-3]))
        diameter=float(min(np.linalg.norm(face[1]-face[0]),np.linalg.norm(face[3]-face[0])))
        yy,xx=np.indices((600,1000));rr=np.hypot(xx-cx,yy-cy)
        visibility={}
        for name,m in [('inner',rr<=.75*r),('scale',(rr>=.82*r)&(rr<=r))]:
            proj=cv2.warpPerspective(m.astype(np.uint8),h,(width,height),flags=cv2.INTER_NEAREST)>0
            visibility[name]=float(np.sum(proj&(owner==index))/np.sum(proj)) if np.any(proj) else 0.
        row.update(face_corners=face.tolist(),face_fits=fits,face_diameter_px=diameter,visibility=visibility,
                   usable_image=bool(fits and diameter>=80 and min(visibility.values())>=.95))
    return image,dict(robot_pose=pose,assets=objects)
