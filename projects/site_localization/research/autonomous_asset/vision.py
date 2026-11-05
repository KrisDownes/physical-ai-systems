"""Pixel-only marker identity/pose and unchanged gauge reader on associated ROI.

No world files, true poses, asset labels or reading reference are loaded here.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


def detect(image,config,inspect=False):
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(dictionary,params)
    corners,ids,_=detector.detectMarkers(image)
    result={'detections':[],'target':None,'identity_decision':'not_observed'}
    if ids is None:return result
    ids=ids.ravel().tolist();result['observed_ids']=ids
    if ids.count(config['target_id'])>1:
        result['identity_decision']='duplicate_id_uncertain';return result
    width,height=config['image_size'];f=width/2/math.tan(math.radians(config['horizontal_fov_deg']/2))
    k=np.array([[f,0,width/2],[0,f,height/2],[0,0,1]],float)
    s=config['marker_size_m']/2
    square=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],np.float32)
    for marker_id,c in zip(ids,corners):
        uv=c[0];edge=float(min(np.linalg.norm(uv[(i+1)%4]-uv[i]) for i in range(4)))
        row=dict(id=marker_id,corners=uv.tolist(),min_edge_px=edge)
        result['detections'].append(row)
        if marker_id!=config['target_id']:continue
        if edge<config['min_marker_edge_px']:
            result['identity_decision']='target_pose_uncertain_small_marker';continue
        from pose import solve_pose
        pose,audit=solve_pose(square,uv,k,config)
        row['pose_audit']=audit
        row['pose_solutions']=audit['usable_solutions']
        if pose is None:
            result['identity_decision']='target_pose_uncertain'
            continue
        # Approximate 1px angular/range sensitivity, not calibrated covariance.
        pose['range_pixel_sensitivity_m']=float(np.linalg.norm(pose['gauge_camera'])/edge)
        result['target']=pose;result['identity_decision']='unique_target_marker_observed'
        if inspect:
            offset=config['marker_to_gauge_right_m'];half=config['reference_width_m']/2
            bounds=np.array([[offset-half,half,0],[offset+half,half,0],
                             [offset+half,-half,0],[offset-half,-half,0]],np.float32)
            projected=cv2.projectPoints(bounds,np.array(pose['rvec']),np.array(pose['tvec']),k,np.zeros(5))[0].reshape(4,2)
            lo=projected.min(axis=0);hi=projected.max(axis=0);margin=.2*(hi-lo)+8
            x0,y0=np.floor(np.maximum([0,0],lo-margin)).astype(int)
            x1,y1=np.ceil(np.minimum([width,height],hi+margin)).astype(int)
            result['gauge_roi_xyxy']=[int(v) for v in [x0,y0,x1,y1]]
            if x1<=x0 or y1<=y0:
                result['reading']={'decision':'abstain','reason':'gauge_roi_outside_image'};continue
            sys.path.insert(0,config['bundle'])
            from commissioned_gauge import CommissionedGauge
            reader=CommissionedGauge(Path(config['bundle']))
            result['reading']=reader.read(image[y0:y1,x0:x1])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cv2.setNumThreads(1)
    result=detect(cv2.imread(str(a.image),cv2.IMREAD_GRAYSCALE),json.loads(a.config.read_text()),a.inspect)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
