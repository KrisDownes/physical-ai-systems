"""Shared visual Cartesian interface for SO-101 contact tasks.

The task name is evaluator/setup metadata, never an input to the learned model.
"""
import numpy as np
import mujoco
from pathlib import Path
from environment import Environment,ROOT,MODEL,Q_HOME,cartesian_step,locate_green,locate_cyan,pixel_to_plane,recalibrate_translation


def slider_model(kind):
    scene=(ROOT/'so101/scene.xml').read_text()
    scene=scene.replace('<freejoint name="object_free"/>','<joint name="object_slide" type="slide" axis="0 1 0" range="-.10 .10" damping=".02"/>')
    # A physical hook/handle, rigidly fixed to the slider; the green body remains visible.
    if kind=='pull':
        scene=scene.replace('</body>\n  </worldbody>','<geom name="slider_handle" type="box" pos="0 0 .019" size=".009 .011 .006" mass=".002" rgba="0 1 .1 1" friction="1 .03 .003"/><geom name="slider_fiducial" type="cylinder" pos=".045 0 .012" size=".008 .002" rgba="0 1 1 1" contype="0" conaffinity="0" mass="0"/>\n    </body>\n  </worldbody>')
    target=MODEL/f'scene_shared_{kind}.xml';target.write_text(scene)
    return mujoco.MjModel.from_xml_path(str(target))

class SharedTask:
    def __init__(self,kind,friction=1.,mass_scale=1.,resistance=.02):
        if kind not in ('push','pull'):raise ValueError(kind)
        self.kind=kind;self.env=Environment(action_budget=250,movement_budget=10.)
        if kind in ('push','pull'):
            self.env.renderer.close();self.env.model=slider_model(kind);self.env.data=mujoco.MjData(self.env.model)
            self.env.nominal_model=slider_model(kind);self.env.nominal_data=mujoco.MjData(self.env.nominal_model)
            self.env.renderer=mujoco.Renderer(self.env.model,240,320)
            self.env.object_id=mujoco.mj_name2id(self.env.model,mujoco.mjtObj.mjOBJ_BODY,'object')
            self.env.site=mujoco.mj_name2id(self.env.model,mujoco.mjtObj.mjOBJ_SITE,'gripperframe')
            self.env.reference_ids=[mujoco.mj_name2id(self.env.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('reference_a','reference_b')]
            self.env.model.dof_frictionloss[6]=resistance
        e=self.env; oid=e.object_id;e.model.body_mass[oid]*=mass_scale;e.model.body_inertia[oid]*=mass_scale
        obj=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,'object_geom');e.model.geom_friction[obj,0]=friction;e.model.geom_priority[obj]=2
        e.model.opt.timestep=.0025
        # Recompute derived subtree mass and inertial constants after changing mass/inertia.
        mujoco.mj_setConst(e.model,e.data)
        original_audit=e._audit_physics_contacts
        def audited_step():
            original_audit()
            now=float(e.data.time)
            if self.audit_time is not None and now+1e-9<self.audit_time:self.numerical_reset=True
            if not np.all(np.isfinite(e.data.qpos)) or not np.all(np.isfinite(e.data.qvel)) or abs(e.data.qpos[6])>.2 or abs(e.data.qvel[6])>20:self.numerical_reset=True
            self.audit_time=now
            self.physics_warnings=[max(v,int(w.number)) for v,w in zip(self.physics_warnings,e.data.warning)]
        e._audit_physics_contacts=audited_step
        self.audit_time=None;self.numerical_reset=False;self.physics_warnings=[0]*7
        self.last=None;self.offset=None;self.last_feature=None;self.frame_count=0

    def reset(self,xy=(.22,.024),seed=0,pan=-.6):
        self.audit_time=None;self.numerical_reset=False;self.physics_warnings=[0]*7
        e=self.env;q=Q_HOME.copy();q[0]=pan
        if self.kind in ('push','pull'):
            e.rng=np.random.default_rng(seed);mujoco.mj_resetData(e.model,e.data)
            e.model.body_pos[e.object_id,:2]=xy;e.data.qpos[:6]=q;e.data.ctrl[:]=q;mujoco.mj_forward(e.model,e.data)
            e.action_count=0;e.movement=0.;e.compute_s=0.;e.trace=[];e.limit_violations=0;e.saturated_actions=0;e.physics_collision_steps=0;e.physics_limit_steps=0;e.physics_collision_pairs={}
            for _ in range(round(.5/e.model.opt.timestep)):mujoco.mj_step(e.model,e.data);e._audit_physics_contacts()
            obs=e.observe()
        self.offset=recalibrate_translation(obs['image'])
        if self.offset is None:raise RuntimeError('calibration unavailable')
        self.last=obs;self.last_feature=None;self.frame_count=0
        return self.features(obs)

    def features(self,obs):
        if self.kind=='pull':
            p=obs['image'];mask=(p[:,:,1]>180)&(p[:,:,2]>180)&(p[:,:,0]<100)
            ys,xs=np.nonzero(mask);pix=np.array([xs.mean(),ys.mean()]) if len(xs)>=5 else None
        else:pix=locate_green(obs['image'])
        visible=pix is not None
        # Missing features are explicitly zero-filled; the visibility mask is supplied.
        xy=pixel_to_plane(pix,z=.034 if self.kind=='pull' else .04,offset_xy=self.offset)[:2] if visible else np.zeros(2)
        if visible and self.kind=='pull':xy[0]-=.045
        q=obs['joint_rad'];tip,_=self.env.controller_fk(q)
        elapsed=.1 if self.last_feature is None else max(obs['time_s']-self.last_feature[0],1e-6)
        if self.last_feature is not None and visible and self.last_feature[2]:vel=(xy-self.last_feature[1])/elapsed
        else:vel=np.zeros(2)
        f=np.r_[xy,vel,tip[:2],q[:6],float(visible)].astype(np.float32)
        self.last_feature=(obs['time_s'],xy.copy(),visible);return f

    def step(self,action,grip):
        action=np.asarray(action,float);action=np.clip(action,-.012,.012)
        self.last=cartesian_step(self.env,self.last,np.r_[action,0.],grip)
        self.last=self.env.observe();self.frame_count+=1
        return self.features(self.last)

    def move_to(self,xyz,grip=1.2,steps=50):
        for _ in range(steps):
            p,_=self.env.controller_fk(self.last['joint_rad']);d=np.asarray(xyz)-p
            if np.linalg.norm(d)<.004:return True
            self.last=cartesian_step(self.env,self.last,d,grip)
        return False

    def truth(self):return self.env.data.xpos[self.env.object_id].copy()
    def close(self):self.env.close()
