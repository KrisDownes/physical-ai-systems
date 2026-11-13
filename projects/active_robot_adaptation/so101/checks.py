"""Model, rendering, timing, reset and contact regression checks."""
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from environment import Environment, locate_green, pixel_to_plane, recalibrate_translation, load_model


def main():
    m=load_model();d=mujoco.MjData(m)
    start=time.perf_counter()
    for _ in range(2000): mujoco.mj_step(m,d)
    duration=time.perf_counter()-start
    elbow=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'elbow_flex')
    collisions={}
    for angle in (1.3,1.47,1.55,1.69):
        mujoco.mj_resetData(m,d);d.qpos[m.jnt_qposadr[elbow]]=angle;mujoco.mj_forward(m,d)
        pairs=[]
        for c in d.contact[:d.ncon]:
            b1,b2=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            if b1 and b2 and m.body_rootid[b1]==m.body_rootid[b2]:
                pairs.append({'bodies':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b1),
                                        mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b2)],
                              'penetration_m':float(-c.dist)})
        collisions[str(angle)]=pairs
    e=Environment()
    obs=e.reset(); first=obs['joint_rad'].copy(); img=obs['image']
    assert img.shape==(240,320,3) and locate_green(img) is not None
    assert np.linalg.norm(pixel_to_plane(locate_green(img))[:2]-[.22,.024]) < .003
    assert np.linalg.norm(recalibrate_translation(img)) < .004
    e.step(np.array([.02,0,0,0,0]), .1)
    moved=e.observe()['joint_rad']
    assert abs(moved[0]-first[0])>.001 and abs(moved[5]-first[5])>.01
    repeated=e.reset()['joint_rad']
    assert np.allclose(first,repeated,atol=1e-12)
    assert set(e.observe().keys())=={'joint_rad','time_s','image'}
    e.close()
    result={'physics_steps':2000,'wall_s':duration,'physics_steps_per_s':2000/duration,
            'image_shape':list(img.shape),'reset_repeatable':True,
            'joint_and_gripper_response':True,'controller_observation_keys':['joint_rad','time_s','image'],
            'self_collision_scan':collisions,'mujoco_version':mujoco.__version__}
    path=Path('results/so101/model_checks.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
