"""SO-101 contact task, agent-authored; evaluator truth is not controller input."""
from pathlib import Path
import time
from collections import deque
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / '.so101-model'
DT = .1
IMAGE_W, IMAGE_H = 320, 240
OBJECT_TOP = .04
CAMERA_HEIGHT = .7
CAMERA_X = .2
FOCAL = IMAGE_H / (2*np.tan(np.deg2rad(45)/2))
Q_GRASP = np.array([0., .000381818, .473496, 1.17717, 1.58437, .727663])
Q_HOME = np.array([-.6, -.35, .50, 1.50, 1.58437, 1.2])


def load_model(tool_marker=False):
    if not (MODEL / 'so101.xml').is_file():
        raise FileNotFoundError('Fetch the pinned model using so101/fetch_model.sh')
    # Scene is copied next to the unmodified upstream model so includes/meshes resolve.
    scene=(ROOT/'so101/scene.xml').read_text()
    if tool_marker:
        original=(MODEL/'so101.xml').read_text()
        needle='<site group="3" name="gripperframe"'
        marker='<geom name="tool_visual_marker" type="sphere" pos="0.012 -0.000218 0.11" size="0.009" rgba="0 0.8 1 1" contype="0" conaffinity="0" mass="0"/>\n                <site name="tool_marker_site" pos="0.012 -0.000218 0.11" size="0.001" rgba="0 0 0 0"/>\n                '
        if original.count(needle)!=1: raise ValueError('unexpected upstream gripper site')
        (MODEL/'so101_observed.xml').write_text(original.replace(needle,marker+needle))
        scene=scene.replace('file="so101.xml"','file="so101_observed.xml"')
    target=MODEL/('scene_task_marked.xml' if tool_marker else 'scene_task.xml')
    target.write_text(scene)
    return mujoco.MjModel.from_xml_path(str(target))


class Environment:
    def __init__(self, camera_shift=(0., 0.), joint_gain=(1,1,1,1,1), render=True, camera_noise_std=0.,
                 tool_marker=False, reference_mode='visible', action_budget=None, movement_budget=None,
                 tool_offset=(0.,0.,0.)):
        self.model = load_model(tool_marker=tool_marker)
        # The controller keeps a distinct unperturbed kinematic model; the fixed
        # wrist-frame mounting displacement moves jaws AND attached marker together.
        self.nominal_model = load_model(tool_marker=tool_marker)
        body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'gripper')
        self.model.body_pos[body] += np.asarray(tool_offset,dtype=float)
        self.data = mujoco.MjData(self.model)
        self.nominal_data = mujoco.MjData(self.nominal_model)
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'overhead')
        self.model.cam_pos[self.cam, :2] += camera_shift
        self.joint_gain = np.asarray(joint_gain)
        self.camera_noise_std = camera_noise_std
        if reference_mode not in ('visible','temporary','unavailable'): raise ValueError(reference_mode)
        self.reference_mode=reference_mode
        self.reference_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('reference_a','reference_b')]
        self.action_budget=action_budget
        self.movement_budget=movement_budget
        self.rng = np.random.default_rng(0)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        self.renderer = mujoco.Renderer(self.model, IMAGE_H, IMAGE_W) if render else None
        self.action_count = 0
        self.movement = 0.
        self.compute_s = 0.
        self._evaluation = {}
        self.trace = []
        self.limit_violations = 0
        self.saturated_actions = 0
        self.physics_collision_steps=0
        self.physics_collision_pairs={}
        self.events=[]

    def reset(self, object_xy=(.22,.024), initial_q=None, seed=0):
        self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        q = Q_HOME.copy() if initial_q is None else np.asarray(initial_q, float).copy()
        q[5] = 1.2
        self.data.qpos[:6] = q
        self.data.qpos[6:9] = [*object_xy, .0201]
        self.data.qpos[9:13] = [1.,0.,0.,0.]
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)
        self.action_count, self.movement, self.compute_s = 0, 0., 0.
        self._evaluation = {'initial_z': float(self.data.xpos[self.object_id,2])}
        self.trace, self.limit_violations, self.saturated_actions = [], 0, 0
        self.physics_collision_steps,self.physics_collision_pairs,self.events=0,{},[]
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
            self._audit_physics_contacts()
        return self.observe()

    def observe(self, include_image=True):
        obs = {'joint_rad': self.data.qpos[:6].copy(), 'time_s': float(self.data.time)}
        if self.renderer is not None and include_image:
            show=self.reference_mode=='visible' or (self.reference_mode=='temporary' and self.data.time>=1.0-1e-9)
            self.model.geom_rgba[self.reference_ids,3]=1. if show else 0.
            self.renderer.update_scene(self.data, camera='overhead')
            obs['image'] = self.renderer.render().copy()
            if self.camera_noise_std:
                noise = self.rng.normal(0.,self.camera_noise_std,obs['image'].shape)
                obs['image'] = np.clip(obs['image'].astype(float)+noise,0,255).astype(np.uint8)
        return obs

    def step(self, delta_joint, gripper):
        """Low-level joint-position wrapper; actual distortion acts on increments."""
        if self.action_budget is not None and self.action_count>=self.action_budget:
            raise RuntimeError('interaction budget exhausted')
        if self.movement_budget is not None and self.movement>=self.movement_budget:
            raise RuntimeError('movement budget exhausted')
        raw = np.asarray(delta_joint)
        action = np.clip(raw, -.08, .08)
        self.saturated_actions += int(np.any(np.abs(raw-action)>1e-12))
        prior = self.data.qpos[:5].copy()
        target = np.clip(prior + action*self.joint_gain,
                         self.model.jnt_range[:5,0], self.model.jnt_range[:5,1])
        self.data.ctrl[:5] = target
        self.data.ctrl[5] = np.clip(gripper, *self.model.jnt_range[5])
        for _ in range(round(DT/self.model.opt.timestep)):
            mujoco.mj_step(self.model,self.data)
            self._audit_physics_contacts()
        self.movement += float(np.linalg.norm(self.data.qpos[:5]-prior))
        self.action_count += 1
        q = self.data.qpos[:6].copy()
        self.limit_violations += int(np.any(q < self.model.jnt_range[:6,0]-.001) or
                                     np.any(q > self.model.jnt_range[:6,1]+.001))
        self.trace.append({'time_s':float(self.data.time),'sent_delta_rad':action.tolist(),
                           'gripper_command_rad':float(gripper),'joint_rad':q.tolist(),
                           'site_xyz_evaluator':self.data.site_xpos[self.site].tolist(),
                           'object_xyz_evaluator':self.data.xpos[self.object_id].tolist()})
        return self.observe(include_image=(self.action_count % 5 == 0))

    def controller_fk(self, q):
        self.nominal_data.qpos[:6] = q
        mujoco.mj_forward(self.nominal_model, self.nominal_data)
        jacp, jacr = np.zeros((3,self.model.nv)), np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.nominal_model,self.nominal_data,jacp,jacr,self.site)
        return self.nominal_data.site_xpos[self.site].copy(), jacp[:,:5].copy()

    def close(self):
        if self.renderer is not None: self.renderer.close()

    def _audit_physics_contacts(self):
        pairs=set()
        for c in self.data.contact[:self.data.ncon]:
            b1,b2=self.model.geom_bodyid[c.geom1],self.model.geom_bodyid[c.geom2]
            if not b1 or not b2 or self.model.body_rootid[b1]!=self.model.body_rootid[b2] or c.dist>=-1e-6: continue
            names=tuple(sorted((mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,b1) or str(b1),
                                mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,b2) or str(b2))))
            # Jaw-to-jaw contact is expected on closure; object has a different root.
            if 'gripper' in names and any(n.startswith('moving_jaw') for n in names): continue
            pairs.add(names)
        if pairs:self.physics_collision_steps+=1
        for pair in pairs:
            k=' / '.join(pair); self.physics_collision_pairs[k]=self.physics_collision_pairs.get(k,0)+1

    def mark_evaluation(self, stage):
        """Evaluator-only snapshot: never returned to the controller during an episode."""
        oid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'object_geom')
        names = []
        for c in self.data.contact[:self.data.ncon]:
            if oid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == oid else c.geom1
                names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other))
        self._evaluation[stage] = {'object_xyz': self.data.xpos[self.object_id].tolist(),
            'site_xyz':self.data.site_xpos[self.site].tolist(),
            'joints_rad':self.data.qpos[:6].tolist(),'contacts': names}

    def evaluate(self):
        e = self._evaluation
        contacts = e.get('after_close', {}).get('contacts', [])
        grasp = all(any((n or '').startswith(prefix) for n in contacts)
                    for prefix in ('fixed_jaw', 'moving_jaw'))
        lifted = e.get('after_lift', {}).get('object_xyz', [0,0,0])[2] > e['initial_z'] + .035
        final = e.get('after_settle', {}).get('object_xyz', [0,0,1])
        placed = np.linalg.norm(np.asarray(final[:2])-[.22,-.05]) < .035 and final[2] < .04
        return {'grasp_contact': grasp, 'lifted': bool(lifted), 'placed': bool(placed),
                'success': bool(grasp and lifted and placed), 'snapshots': e}


def locate_green(image):
    """Estimate visible green object center from RGB pixels; return None if occluded."""
    p = image.astype(int)
    mask = (p[:,:,1] > 100) & (p[:,:,1] > p[:,:,0]*1.7) & (p[:,:,1] > p[:,:,2]*1.5)
    ys,xs = np.nonzero(mask)
    if len(xs) < 8: return None
    return np.array([xs.mean(),ys.mean()])


def locate_cyan(image):
    p=image.astype(int)
    mask=(p[:,:,1]>220)&(p[:,:,2]>220)&(p[:,:,0]<90)
    ys,xs=np.nonzero(mask)
    return None if len(xs)<5 else np.array([xs.mean(),ys.mean()])


def pixel_to_plane(pixel, z=OBJECT_TOP, offset_xy=(0.,0.)):
    u,v = pixel
    depth = CAMERA_HEIGHT-z
    return np.array([CAMERA_X+(u-IMAGE_W/2)*depth/FOCAL + offset_xy[0],
                     -(v-IMAGE_H/2)*depth/FOCAL + offset_xy[1], z])


def recalibrate_translation(image):
    """Estimate overhead camera translation from two declared magenta table fiducials."""
    p = image.astype(int)
    mask = (p[:,:,0] > 100) & (p[:,:,2] > 100) & (p[:,:,1] < 100)
    ys,xs = np.nonzero(mask)
    if len(xs) < 12: return None
    split = (xs.min()+xs.max())/2
    groups = [xs < split, xs >= split]
    if min(int(g.sum()) for g in groups) < 6: return None
    pixels = [np.array([xs[g].mean(),ys[g].mean()]) for g in groups]
    refs = [np.array([.10,-.17]),np.array([.32,.17])]
    return np.mean([ref-pixel_to_plane(px,z=.002)[:2] for px,ref in zip(pixels,refs)],axis=0)


class OnlineJacobian:
    """RLS estimate of command-to-observed-joint response, composed with nominal FK."""
    def __init__(self):
        self.b = np.eye(5)
        self.p = np.eye(5)*1000.

    def update(self, command, q_before, q_after):
        if np.linalg.norm(command) < .002: return False
        pu = self.p @ command
        k = pu/(.98+command@pu)
        self.b += np.outer(q_after-q_before-self.b@command,k)
        self.p = (self.p-np.outer(k,command@self.p))/.98
        return True


class GlobalCorrection:
    """Local FK-predicted vs observed displacement RLS; optional GP residual."""
    def __init__(self, gp=False):
        self.b=np.eye(3)
        self.p=np.eye(3)*1000.
        self.data=deque(maxlen=16)
        self.gp=gp

    def correct(self, desired, position):
        residual=np.zeros(3)
        if self.gp and len(self.data)>=4:
            z=np.array([np.r_[p/.2,d/.02] for p,d,_ in self.data])
            y=np.array([observed-self.b@d for _,d,observed in self.data])
            dist=((z[:,None,:]-z[None,:,:])**2).sum(axis=2)
            kernel=np.exp(-dist/(2*1.2**2))
            query=np.r_[position/.2,desired/.02]
            cross=np.exp(-((z-query)**2).sum(axis=1)/(2*1.2**2))
            residual=cross@np.linalg.solve(kernel+.0015*np.eye(len(z)),y)
        return np.linalg.solve(self.b.T@self.b+.02**2*np.eye(3), self.b.T@(desired-residual))

    def update(self, position, predicted, actual):
        if np.linalg.norm(predicted)<.0003:return False
        pz=self.p@predicted
        k=pz/(.97+predicted@pz)
        self.b+=np.outer(actual-self.b@predicted,k)
        self.p=(self.p-np.outer(k,predicted@self.p))/.97
        self.data.append((position.copy(),predicted.copy(),actual.copy()))
        return True


def task_jacobian(j, response=None):
    """Apply the same joint-response estimate to translation and pitch proxy."""
    task=np.vstack([j, np.array([0.,.10,.10,.10,0.])])
    return (task if response is None else task@response)[:,:4]


def cartesian_step(env, obs, delta, gripper, online=None, global_model=None):
    """Translate in world metres with fixed wrist joints; no arbitrary 6-DoF pose claim."""
    start=time.perf_counter()
    position, j = env.controller_fk(obs['joint_rad'])
    nominal_j=j.copy()
    # Regulate a feasible pitch proxy while translating; wrist roll stays fixed.
    j4 = task_jacobian(j,None if online is None else online.b)
    pitch_error = .10 * ((Q_GRASP[1:4].sum()) - obs['joint_rad'][1:4].sum())
    delta = np.asarray(delta, float)
    delta *= min(1., .015 / max(np.linalg.norm(delta), 1e-12))
    if global_model is not None:
        delta=global_model.correct(delta,position)
    goal = np.r_[delta, np.clip(pitch_error,-.02,.02)]
    dq = j4.T @ np.linalg.solve(j4@j4.T + .02**2*np.eye(4), goal)
    env.compute_s += time.perf_counter()-start
    u = np.r_[dq,0.]
    after = env.step(u, gripper)
    if online is not None:
        start=time.perf_counter()
        prior=online.b.copy()
        accepted=online.update(np.clip(u,-.08,.08),obs['joint_rad'][:5],after['joint_rad'][:5])
        env.trace[-1]['adaptation']={'kind':'online','accepted':accepted,'input_u':np.clip(u,-.08,.08).tolist(),
            'reason':'accepted' if accepted else 'command_too_small',
            'predicted_delta_q':(prior@np.clip(u,-.08,.08)).tolist(),
            'observed_delta_q':(after['joint_rad'][:5]-obs['joint_rad'][:5]).tolist(),'b':online.b.tolist()}
        env.compute_s += time.perf_counter()-start
    if global_model is not None:
        start=time.perf_counter()
        new_position,_=env.controller_fk(after['joint_rad'])
        predicted=nominal_j@np.clip(u,-.08,.08);actual=new_position-position
        accepted=global_model.update(position,predicted,actual)
        env.trace[-1]['adaptation']={'kind':'gp' if global_model.gp else 'global','accepted':accepted,
            'input_predicted_xyz':predicted.tolist(),'observed_fk_delta_xyz':actual.tolist(),
            'dataset_size':len(global_model.data),'b':global_model.b.tolist()}
        env.compute_s+=time.perf_counter()-start
    return after
