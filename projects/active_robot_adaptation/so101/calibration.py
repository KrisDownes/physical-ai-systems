"""Restricted camera-translation calibration from declared image features."""
import numpy as np
import mujoco
from environment import locate_cyan,pixel_to_plane


def table_offsets(image):
    p=image.astype(int)
    mask=(p[:,:,0]>100)&(p[:,:,2]>100)&(p[:,:,1]<100)
    ys,xs=np.nonzero(mask)
    if len(xs)<12:return None
    split=(xs.min()+xs.max())/2
    groups=[xs<split,xs>=split]
    if min(int(g.sum()) for g in groups)<6:return None
    refs=[np.array([.10,-.17]),np.array([.32,.17])]
    return np.array([ref-pixel_to_plane([xs[g].mean(),ys[g].mean()],z=.002)[:2]
                     for g,ref in zip(groups,refs)])


def blue_destination(image):
    p=image.astype(int)
    mask=(p[:,:,2]>180)&(p[:,:,2]>p[:,:,1]*1.12)&(p[:,:,2]>p[:,:,0]*1.25)
    y,x=np.nonzero(mask)
    return None if len(x)<10 else np.array([x.mean(),y.mean()])


def select_calibration(env,obs):
    """Prefer consistent table references; marker-only fallback is conditional."""
    log=[]; estimates=table_offsets(obs['image']); table=None; table_invalid=False
    if estimates is not None:
        if np.linalg.norm(estimates[0]-estimates[1])<=.006 and np.linalg.norm(estimates.mean(axis=0))<=.04:
            table=estimates.mean(axis=0)
            log.append({'source':'table','accepted':True,'reason':'consistent_references_within_bound','offset_xy':table.tolist()})
        else:
            table_invalid=True
            log.append({'source':'table','accepted':False,'reason':'inconsistent_or_out_of_range'})
    else: log.append({'source':'table','accepted':False,'reason':'references_missing'})
    pix=locate_cyan(obs['image']); marker=None
    if pix is not None:
        env.controller_fk(obs['joint_rad'])
        idx=mujoco.mj_name2id(env.nominal_model,mujoco.mjtObj.mjOBJ_SITE,'tool_marker_site')
        if idx>=0:
            pos=env.nominal_data.site_xpos[idx].copy()
            marker=pos[:2]-pixel_to_plane(pix,z=pos[2])[:2]
    if marker is None: log.append({'source':'gripper','accepted':False,'reason':'marker_missing'})
    elif table is not None:
        diff=float(np.linalg.norm(marker-table))
        log.append({'source':'gripper','accepted':False,'reason':'inconsistent_with_table' if diff>.0045 else 'table_preferred',
                    'difference_m':diff,'offset_xy':marker.tolist()})
    elif table_invalid:
        log.append({'source':'gripper','accepted':False,'reason':'table_geometry_invalid'})
    elif np.linalg.norm(marker)<=.04:
        log.append({'source':'gripper','accepted':True,'reason':'fallback_assumes_nominal_geometry','offset_xy':marker.tolist()})
    else: log.append({'source':'gripper','accepted':False,'reason':'out_of_range'})
    return (table if table is not None else marker if not table_invalid and marker is not None and np.linalg.norm(marker)<=.04 else None),log


class CalibrationState:
    """Joint-space RLS survives calibration; Cartesian residual state is invalidated."""
    def __init__(self): self.offset=None; self.epoch=0; self.events=[]
    def apply(self,offset,joint_model=None,cartesian_model=None):
        offset=np.asarray(offset,dtype=float)
        changed=self.offset is None or not np.allclose(offset,self.offset,atol=1e-9,rtol=0)
        if changed:
            self.epoch+=1
            if cartesian_model is not None:
                cartesian_model.b=np.eye(3);cartesian_model.p=np.eye(3)*1000.;cartesian_model.data.clear()
            self.events.append({'epoch':self.epoch,'offset_xy':offset.tolist(),
              'joint_response_state':'retained_joint_frame' if joint_model is not None else 'absent',
              'cartesian_residual_state':'invalidated' if cartesian_model is not None else 'absent'})
        self.offset=offset.copy()


class VisualTargets:
    """Recompute both targets from retained image observations after calibration."""
    def __init__(self,object_pixel,destination_pixel):
        self.object_pixel=np.asarray(object_pixel).copy()
        self.destination_pixel=np.asarray(destination_pixel).copy()
    def reproject(self,offset):
        return (pixel_to_plane(self.object_pixel,offset_xy=offset),
                pixel_to_plane(self.destination_pixel,z=.002,offset_xy=offset)[:2])
