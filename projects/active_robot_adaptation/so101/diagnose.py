"""Paired camera-failure tracing and explicitly oracle-labeled causal interventions."""
import json
import time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from environment import Environment,Q_HOME,locate_green,FOCAL,IMAGE_W,IMAGE_H
from nominal import run
from run_store import create_run

CASE={'object_xy':[.217,.021],'initial_pan':-.58,'seed':701}
CONDITIONS=[('nominal','nominal',[0,0],None),('shift_nominal','nominal',[.012,-.008],None),
            ('shift_online','online',[.012,-.008],None),('shift_global','global',[.012,-.008],None),
            ('shift_gp','gp',[.012,-.008],None),
            ('oracle_pixel','nominal',[.012,-.008],'oracle_pixel'),
            ('oracle_target_nominal','nominal',[.012,-.008],'oracle_target'),
            ('oracle_target_online','online',[.012,-.008],'oracle_target'),
            ('shift_selfcal','selfcal',[.012,-.008],None)]


def project_truth(point,shift):
    # Evaluator-only overlay using the actual, known simulation camera pose.
    depth=.7-point[2]
    return (IMAGE_W/2+FOCAL*(point[0]-.2-shift[0])/depth,
            IMAGE_H/2-FOCAL*(point[1]-shift[1])/depth)


def overlay(frames,shift,path):
    out=[]
    for d in frames:
        image=Image.fromarray(d['image']);draw=ImageDraw.Draw(image)
        obj=locate_green(d['image']); grip=d.get('gripper_pixel')
        if obj is not None:
            x,y=obj;draw.ellipse((x-5,y-5,x+5,y+5),outline='lime',width=2)
        if grip is not None:
            x,y=grip;draw.ellipse((x-6,y-6,x+6,y+6),outline='cyan',width=2)
        if 'site_xyz' in d:
            x,y=project_truth(d['site_xyz'],shift)
            draw.line((x-5,y,x+5,y),fill='deepskyblue',width=2)
            draw.line((x,y-5,x,y+5),fill='deepskyblue',width=2)
        if 'target_xyz' in d:
            x,y=project_truth(d['target_xyz'],shift)
            draw.line((x-6,y,x+6,y),fill='red',width=2);draw.line((x,y-6,x,y+6),fill='red',width=2)
        draw.rectangle((0,0,320,30),fill='black')
        draw.text((3,2),d['stage']+' green=object cyan=marker',fill='white')
        draw.text((3,15),'red=target* blue=tip* (*evaluator projection)',fill='white')
        out.append(np.asarray(image))
    imageio.mimsave(path,out,duration=200,loop=0)


def main():
    out=create_run('camera_diagnosis',{'case':CASE,'conditions':CONDITIONS,
        'tool_marker':'non-colliding simulated cyan marker on a 0.11 m mast; not hardware validated',
        'seed':CASE['seed']})
    rows=[]
    for name,method,shift,intervention in CONDITIONS:
        e=Environment(camera_shift=shift,tool_marker=True)
        q=Q_HOME.copy();q[0]=CASE['initial_pan']
        start=time.perf_counter()
        r=run(e,CASE['object_xy'],q,method=method,seed=CASE['seed'],intervention=intervention,diagnostics=True)
        elapsed=time.perf_counter()-start
        frames=r.pop('diagnostic_frames',[]);r.pop('frames',None)
        if frames:
            overlay(frames,shift,out/f'{name}_overlay.gif')
            r['evaluator_initial_object_xyz']=frames[0]['object_xyz']
            r['evaluator_initial_gripper_site_xyz']=frames[0]['site_xyz']
            r['visual_samples']=[{k:v for k,v in d.items() if k!='image'} for d in frames]
        accepted=[t['adaptation'] for t in e.trace if 'adaptation' in t and t['adaptation']['accepted']]
        r.update(name=name,method=method,shift=shift,intervention=intervention,actions=e.action_count,
                 time_s=float(e.data.time),wall_s=elapsed,collision_steps=e.physics_collision_steps,
                 collision_pairs=e.physics_collision_pairs,accepted_updates=len(accepted),trace=e.trace)
        rows.append(r);e.close()
        print(name,'success',r['success'],'failure',r['failure'],'updates',len(accepted),flush=True)
    (out/'diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('saved:',out)


if __name__=='__main__':main()
