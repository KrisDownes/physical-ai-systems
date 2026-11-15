"""Small rigid bottle task; mass/friction and all physics telemetry are evaluator-only."""
from pathlib import Path
import math
import numpy as np
import mujoco
from environment import Environment, MODEL, ROOT, IMAGE_W, IMAGE_H

RADIUS=.013
HEIGHT=.040
TARE=.012
INNER_RADIUS=.012
INNER_HEIGHT=.038
DENSITY=1000.
FILL={'empty':0.,'half':.5,'full':1.}
FRICTION={'dry':1.,'wet':.3}  # hypothesized proxies, not measured wet-contact properties

def physical_parameters(fill):
    fraction=FILL[fill]; water=DENSITY*math.pi*INNER_RADIUS**2*INNER_HEIGHT*fraction
    # Uniform solid cylindrical tare and bottom-filled water cylinder; both centered on axis.
    z_water=-HEIGHT/2+.001+(INNER_HEIGHT*fraction)/2 if water else 0.
    total=TARE+water; z=water*z_water/total
    it_xy=TARE*(3*RADIUS**2+HEIGHT**2)/12 + TARE*z*z
    iw_xy=water*(3*INNER_RADIUS**2+(INNER_HEIGHT*fraction)**2)/12 + water*(z_water-z)**2
    iz=.5*TARE*RADIUS**2+.5*water*INNER_RADIUS**2
    return {'mass_kg':total,'water_kg':water,'com_z_m':z,'inertia_kg_m2':[it_xy+iw_xy,it_xy+iw_xy,iz]}

def bottle_model(fill):
    # Single opaque cylindrical bottle: analytical inertia + explicit free body; no weld.
    params=physical_parameters(fill);scene=(ROOT/'so101/scene.xml').read_text()
    old='<geom name="object_geom" type="box" size=".014 .014 .02" mass=".02" rgba="0 1 .1 1" friction="1 .03 .003" condim="6"/>'
    inertia=' '.join(f'{x:.12g}' for x in params['inertia_kg_m2'])
    new=(f'<inertial pos="0 0 {params["com_z_m"]:.12g}" mass="{params["mass_kg"]:.12g}" diaginertia="{inertia}"/>'
         '<geom name="object_geom" type="cylinder" size=".013 .020" rgba="0 1 .1 1" friction=".3 .005 .0005" priority="1" condim="6"/>')
    assert scene.count(old)==1
    scene=scene.replace(old,new).replace('file="so101.xml"','file="so101_observed.xml"')
    # Temporary compiled-input XML is not experiment evidence; full generator and snapshot are saved.
    target=MODEL/'scene_bottle_generated.xml';target.write_text(scene)
    return mujoco.MjModel.from_xml_path(str(target))

class BottleEnvironment(Environment):
    def __init__(self,fill,grip,physics_timestep=None,**kwargs):
        super().__init__(tool_marker=True,**kwargs)
        if self.renderer is not None:self.renderer.close()
        self.model=bottle_model(fill);self.data=mujoco.MjData(self.model)
        if physics_timestep is not None:self.model.opt.timestep=physics_timestep
        self.renderer=mujoco.Renderer(self.model,IMAGE_H,IMAGE_W) if kwargs.get('render',True) else None
        self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'gripperframe')
        self.object_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'object')
        self.cam=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,'overhead')
        self.reference_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('reference_a','reference_b')]
        self.geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
        self.gripper_body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'gripper')
        for i in range(self.model.ngeom):
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
            body=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[i])) or ''
            if self.model.geom_contype[i] and (name.startswith(('fixed_jaw','moving_jaw')) or body in ('gripper','moving_jaw_so101_v1')):
                self.model.geom_friction[i,0]=FRICTION[grip]
        self.telemetry={}

    def reset(self,*args,**kwargs):
        self.telemetry={'peak_normal_N':0.,'normal_impulse_Ns':0.,'max_tilt_deg':0.,'transport_table_contact_steps':0,
                        'max_actuator_effort_Nm':[0.]*6,'actuator_abs_impulse_Nms':[0.]*6,'jaw_contact_steps':0,
                        'friction_observed':[],'max_grip_relative_motion_m':0.,'max_joint_tracking_error_rad':0.,'force_saturation_steps':0}
        self.grasp_relative=None;self.phase='reset'
        return super().reset(*args,**kwargs)

    def record_phase(self,phase):self.phase=phase

    def _relative(self):
        rotation=self.data.xmat[self.gripper_body].reshape(3,3)
        return rotation.T@(self.data.xpos[self.object_id]-self.data.site_xpos[self.site])

    def _audit_physics_contacts(self):
        super()._audit_physics_contacts()
        if not self.telemetry:return
        dt=self.model.opt.timestep;forces=np.abs(self.data.actuator_force[:6]);m=self.telemetry
        m['max_actuator_effort_Nm']=np.maximum(m['max_actuator_effort_Nm'],forces).tolist()
        m['actuator_abs_impulse_Nms']=(np.asarray(m['actuator_abs_impulse_Nms'])+forces*dt).tolist()
        m['force_saturation_steps']+=int(np.any(forces>=2.94-1e-5))
        m['max_joint_tracking_error_rad']=max(m['max_joint_tracking_error_rad'],float(np.max(np.abs(self.data.ctrl[:6]-self.data.qpos[:6]))))
        force=np.zeros(6);jaw=False;table=False
        for i,c in enumerate(self.data.contact[:self.data.ncon]):
            if self.geom not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.geom else c.geom1
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,other) or ''
            body=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[other])) or ''
            mujoco.mj_contactForce(self.model,self.data,i,force)
            if name.startswith(('fixed_jaw','moving_jaw')) or body in ('gripper','moving_jaw_so101_v1'):
                jaw=True;normal=max(0.,float(force[0]));m['peak_normal_N']=max(m['peak_normal_N'],normal);m['normal_impulse_Ns']+=normal*dt
                f=round(float(c.friction[0]),6)
                if f not in m['friction_observed']:m['friction_observed'].append(f)
            if name=='table' and self.phase=='transport':table=True
        m['jaw_contact_steps']+=int(jaw)
        m['transport_table_contact_steps']+=int(table)
        if self.grasp_relative is not None and self.data.ctrl[5]<.5:
            m['max_grip_relative_motion_m']=max(m['max_grip_relative_motion_m'],float(np.linalg.norm(self._relative()-self.grasp_relative)))
            m['max_tilt_deg']=max(m['max_tilt_deg'],float(np.degrees(np.arccos(np.clip(self.data.xmat[self.object_id].reshape(3,3)[2,2],-1,1)))))

    def mark_evaluation(self,stage):
        super().mark_evaluation(stage)
        for c in self.data.contact[:self.data.ncon]:
            if self.geom not in (c.geom1,c.geom2):continue
            other=c.geom2 if c.geom1==self.geom else c.geom1
            body=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[other])) or ''
            if body=='gripper':self._evaluation[stage]['contacts'].append('fixed_jaw_body')
            if body=='moving_jaw_so101_v1':self._evaluation[stage]['contacts'].append('moving_jaw_body')
        if stage=='after_close':self.grasp_relative=self._relative().copy()
        snap=self._evaluation[stage];snap['tilt_deg']=float(np.degrees(np.arccos(np.clip(self.data.xmat[self.object_id].reshape(3,3)[2,2],-1,1))))
        snap['linear_speed_m_s']=float(np.linalg.norm(self.data.qvel[6:9]));snap['angular_speed_rad_s']=float(np.linalg.norm(self.data.qvel[9:12]))
        snap['gripper_relative_xyz_m']=self._relative().tolist()

    def step(self,*args,**kwargs):
        obs=super().step(*args,**kwargs)
        self.trace[-1]['bottle_relative_gripper_evaluator_m']=self._relative().tolist()
        self.trace[-1]['bottle_tilt_deg_evaluator']=float(np.degrees(np.arccos(np.clip(self.data.xmat[self.object_id].reshape(3,3)[2,2],-1,1))))
        self.trace[-1]['actuator_force_evaluator_Nm']=self.data.actuator_force[:6].tolist()
        return obs

    def evaluate(self):
        result=super().evaluate();s=result['snapshots'];final=s.get('after_settle',{})
        upright=bool(final and final['tilt_deg']<15.)
        stable=bool(final and final['linear_speed_m_s']<.015 and final['angular_speed_rad_s']<.2 and abs(final['object_xyz'][2]-.02)<.005)
        result['upright']=upright;result['stable']=stable
        result['success']=bool(result['success'] and upright and stable)
        return result
