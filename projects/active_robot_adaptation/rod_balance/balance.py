"""Independent idealized release on a fixed support, using actual MuJoCo contact."""
import math
import numpy as np
import mujoco

def evaluate(theta, choice, render=False):
 # Local y is rod length. Choice is location along rod put over support centre.
 # The two inertial masses and rod dimensions match the ASID XML; this scene is
 # an independent idealized placement diagnostic, not an end-to-end robot.
 xml=f'''<mujoco><option timestep="0.002" gravity="0 0 -9.81"/><worldbody>
 <geom name="floor" type="plane" pos="0 0 -0.15" size="1 1 .1"/>
 <geom name="support" type="box" pos="0 0 -0.01" size=".04 .012 .01" rgba=".2 .6 .2 1"/>
 <body name="rod" pos="0 {-choice} .0205"><freejoint/>
 <inertial pos="0 0 0" mass=".01" diaginertia=".0002 .0002 .0002"/>
 <geom name="rod_geom" type="box" size=".02 .15 .02" mass="0" rgba=".6 .2 .2 1"/>
 <body pos="0 {theta} 0"><inertial pos="0 0 0" mass=".015" diaginertia=".0002 .0002 .0002"/></body>
 </body></worldbody></mujoco>'''
 m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);frames=[];renderer=mujoco.Renderer(m,240,320) if render else None
 support_id=m.geom('support').id;rod_id=m.geom('rod_geom').id; stable=[]
 for step in range(1000):
  mujoco.mj_step(m,d)
  if step>=750:
   rot=d.xmat[m.body('rod').id].reshape(3,3)
   touching=any({int(c.geom1),int(c.geom2)}=={support_id,rod_id} for c in d.contact[:d.ncon])
   stable.append(abs(rot[2,1])<math.sin(.1) and np.linalg.norm(d.qvel[:3])<.05 and np.linalg.norm(d.qvel[3:])<.5 and touching)
  if render and step%100==0:
   renderer.update_scene(d);frames.append(renderer.render().copy())
 rot=d.xmat[m.body('rod').id].reshape(3,3);tilt=math.asin(min(1,abs(rot[2,1])));speed=np.linalg.norm(d.qvel[:3]);angular=np.linalg.norm(d.qvel[3:]);support_id=m.geom('support').id;rod_id=m.geom('rod_geom').id
 contact=any({int(c.geom1),int(c.geom2)}=={support_id,rod_id} for c in d.contact[:d.ncon])
 success=all(stable)
 if renderer:renderer._gl_context.free()
 return {'success':bool(success),'stability_window_s':.5,'stable_steps':int(sum(stable)),'tilt_rad':float(tilt),'linear_speed_m_s':float(speed),'angular_speed_rad_s':float(angular),'support_contact':bool(contact),'final_center':d.qpos[:3].tolist(),'placement_error_evaluator_m':float(.6*theta-choice),'simulation_s':2.,'warnings':[int(w.number) for w in d.warning]},frames

if __name__=='__main__':
 for t in [-.1,0,.1]:
  for c in [0,.6*t]:print(t,c,evaluate(t,c)[0])
