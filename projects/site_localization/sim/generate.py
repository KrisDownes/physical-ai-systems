"""Small commissioned development cell; analytic prior map, not a SLAM claim.

Robot uses Gazebo's differential-drive, odometry, contact and lidar systems.
Geometry is project-generated. This is a development fixture, never a holdout.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def add(parent, tag, text=None, **attrs):
    e=ET.SubElement(parent,tag,attrs)
    if text is not None:e.text=str(text)
    return e


def box(parent,name,x,y,sx,sy,z=.5,sz=1):
    m=add(parent,'model',name=name);add(m,'static','true');add(m,'pose',f'{x} {y} {z} 0 0 0')
    l=add(m,'link',name='link')
    for tag in ['collision','visual']:
        g=add(add(l,tag,name=tag),'geometry');add(add(g,'box'),'size',f'{sx} {sy} {sz}')


def main():
    root=Path(__file__).resolve().parent
    sdf=ET.Element('sdf',version='1.9');w=add(sdf,'world',name='site_cell')
    ph=add(w,'physics',name='cpu',type='ignored');add(ph,'max_step_size',.002);add(ph,'real_time_factor',1)
    add(w,'gravity','0 0 -9.81')
    for lib,cls in [('physics','Physics'),('sensors','Sensors'),('contact','Contact'),('scene-broadcaster','SceneBroadcaster')]:
        p=add(w,'plugin',filename=f'gz-sim-{lib}-system',name=f'gz::sim::systems::{cls}')
        if cls=='Sensors':add(p,'render_engine','ogre2')
    box(w,'floor',0,0,8,8,-.05,.1)
    obstacles=[[-3.2,0,.15,6.55],[3.2,0,.15,6.55],[0,-3.2,6.55,.15],[0,3.2,6.55,.15],
               [0,.4,.8,.6],[-2.3,1,.5,.6],[2.5,2,.4,.9]]
    for i,o in enumerate(obstacles):box(w,f'obstacle_{i}',*o)
    r=add(w,'model',name='rover');add(r,'pose','-2 -2 0 0 0 0')
    body=add(r,'link',name='body');add(body,'pose','0 0 .16 0 0 0')
    inert=add(body,'inertial');add(inert,'mass',2)
    it=add(inert,'inertia')
    for k in ['ixx','iyy','izz']:add(it,k,.03)
    for kind in ['collision','visual']:
        g=add(add(body,kind,name='body_'+kind),'geometry');add(add(g,'box'),'size','.35 .28 .12')
    sensor=add(body,'sensor',name='scan',type='gpu_lidar');add(sensor,'pose','0 0 .2 0 0 0')
    add(sensor,'topic','/site/scan');add(sensor,'update_rate',10);add(sensor,'always_on','true')
    lidar=add(sensor,'lidar');horizontal=add(add(lidar,'scan'),'horizontal')
    for k,v in [('samples',181),('resolution',1),('min_angle',-np.pi),('max_angle',np.pi)]:add(horizontal,k,v)
    ran=add(lidar,'range')
    for k,v in [('min',.05),('max',8),('resolution',.01)]:add(ran,k,v)
    contact=add(body,'sensor',name='body_contact',type='contact');add(contact,'always_on','true');add(contact,'update_rate',20)
    contact_spec=add(contact,'contact');add(contact_spec,'collision','body_collision');add(contact_spec,'topic','/site/contacts')
    for side,y in [('left',.19),('right',-.19)]:
        l=add(r,'link',name=side);add(l,'pose',f'0 {y} .1 {-np.pi/2} 0 0')
        inert=add(l,'inertial');add(inert,'mass',.2);it=add(inert,'inertia')
        for k in ['ixx','iyy','izz']:add(it,k,.001)
        for kind in ['collision','visual']:
            e=add(l,kind,name=kind);g=add(e,'geometry');c=add(g,'cylinder');add(c,'radius',.1);add(c,'length',.035)
        if side=='left':
            sensor=add(l,'sensor',name='wheel_contact',type='contact');add(sensor,'always_on',1);add(sensor,'update_rate',20)
            cs=add(sensor,'contact');add(cs,'collision','collision');add(cs,'topic','/judge/wheel_contacts')
        j=add(r,'joint',name=side+'_joint',type='revolute');add(j,'parent','body');add(j,'child',side)
        a=add(j,'axis');add(a,'xyz','0 1 0',expressed_in='__model__')
    for name,x in [('front',.14),('back',-.14)]:
        l=add(r,'link',name=name);add(l,'pose',f'{x} 0 .05 0 0 0')
        inert=add(l,'inertial');add(inert,'mass',.05);it=add(inert,'inertia')
        for k in ['ixx','iyy','izz']:add(it,k,.0001)
        for kind in ['collision','visual']:
            e=add(l,kind,name=kind);g=add(e,'geometry');add(add(g,'sphere'),'radius',.05)
            if kind=='collision':
                ode=add(add(add(e,'surface'),'friction'),'ode');add(ode,'mu',0);add(ode,'mu2',0)
        j=add(r,'joint',name=name+'_joint',type='fixed');add(j,'parent','body');add(j,'child',name)
    drive=add(r,'plugin',filename='gz-sim-diff-drive-system',name='gz::sim::systems::DiffDrive')
    for k,v in [('left_joint','left_joint'),('right_joint','right_joint'),('wheel_separation',.38),('wheel_radius',.1),
                ('topic','/site/cmd_vel'),('odom_topic','/site/odom'),('odom_publish_frequency',30)]:add(drive,k,v)
    truth=add(r,'plugin',filename='gz-sim-odometry-publisher-system',name='gz::sim::systems::OdometryPublisher')
    for k,v in [('odom_topic','/judge/truth'),('odom_publish_frequency',30),('dimensions',2)]:add(truth,k,v)
    ET.indent(sdf);ET.ElementTree(sdf).write(root/'cell.sdf',encoding='unicode')
    # Perfect commissioned geometric prior; final experiments must also test acquired maps.
    resolution=.05; width=140;origin=-3.5
    xy=origin+np.arange(width)*resolution
    x,y=np.meshgrid(xy,xy);grid=np.full((width,width),-1)
    for ox,oy,sx,sy in obstacles:grid[(abs(x-ox)<=sx/2)&(abs(y-oy)<=sy/2)]=1
    with (root/'map.txt').open('w') as f:
        f.write(f'{width} {width} {resolution} {origin} {origin}\n');np.savetxt(f,grid,fmt='%d')
    cfg=dict(split='development',start=[-2,-2,0],goals=[[-1.5,-1.5,0],[1.5,-1.5,float(np.pi/2)],
         [1.5,1.5,float(np.pi)],[-1.5,1.5,float(-np.pi/2)]],dwell_s=2,
         judge_position_m=.15,judge_yaw_rad=float(np.deg2rad(10)),max_sim_s=150,
         obstacles=obstacles,map_source='analytic commissioned geometry; privileged prior',
         sensor_noise='none',scenario='unchanged_development_cell')
    (root/'cell.json').write_text(json.dumps(cfg,indent=2)+'\n')


if __name__=='__main__':main()
