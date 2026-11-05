"""Predeclared synthetic component checks; evaluation truth never enters detect()."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from vision import detect
from pose import solve_pose

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research/moved_asset'))
from render import render


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'results/moved_asset_20261024/v2/unchanged__autonomous'
    config=json.loads((old/'public/vision.json').read_text());scene=json.loads((old/'private/scene.json').read_text())
    scene['obstacles']=[];texture=scene['assets'][0]['texture'];scene['assets']=[dict(id=7,pose=[0,0,0],texture=texture)]
    specifications=[('front',[-1.21,.08,0],'valid'),('oblique',[-1.4,.38,-.18],'valid'),
        ('wrong_id',[-1.13,.04,0],'reject'),('duplicate_id',[-2,0,0],'reject'),
        ('empty',[-1.13,.04,0],'reject'),('small_marker',[-20,0,0],'reject')]
    (out/'specifications.json').write_text(json.dumps(specifications,indent=2)+'\n');results=[]
    for name,robot,expected in specifications:
        sc=json.loads(json.dumps(scene))
        if name=='wrong_id':sc['assets'][0].update(id=8,texture=texture.replace('asset_7','asset_8'))
        if name=='duplicate_id':sc['assets']=[dict(id=7,pose=[0,-.5,0],texture=texture),dict(id=7,pose=[0,.5,0],texture=texture)]
        if name=='empty':sc['assets']=[]
        image,truth=render(robot,sc);cv2.imwrite(str(out/(name+'.png')),image)
        result=detect(image,config);pose=result.get('target');error=None
        if pose and expected=='valid':
            dx,dy=-robot[0],-robot[1];a=robot[2]
            actual=np.array([math.sin(a)*dx-math.cos(a)*dy,0,math.cos(a)*dx+math.sin(a)*dy])
            error=float(np.linalg.norm(actual-pose['gauge_camera']))
        passed=bool(pose is not None and error<=.05) if expected=='valid' else pose is None
        results.append(dict(name=name,expected=expected,decision=result['identity_decision'],relative_error_m=error,passed=passed))
        (out/(name+'.json')).write_text(json.dumps(dict(method=result,evaluation=truth),indent=2)+'\n')
    s=.08;obj=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]])
    a=np.deg2rad(11);rot=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])@np.diag([1.,-1.,-1.])
    f=960/math.tan(math.pi/6);k=np.array([[f,0,960],[0,f,540],[0,0,1.]])
    uv=cv2.projectPoints(obj,cv2.Rodrigues(rot)[0],np.array([0.,0.,4.7]),k,None)[0].reshape(4,2)
    for name,points in [('ambiguous',uv),('invalid_corners',uv+np.array([[30.,0],[0,0],[0,0],[0,0]]))]:
        pose,audit=solve_pose(obj,points,k,config)
        (out/(name+'.json')).write_text(json.dumps(dict(corners=points.tolist(),audit=audit),indent=2)+'\n')
        results.append(dict(name=name,expected='reject',decision=audit['decision'],passed=pose is None))
    replay=ROOT/'results/acquired_asset_20261024/v1/wrong_asset__supplied/camera/observation_016.png'
    r=detect(cv2.imread(str(replay),0),config)
    (out/'original_regression.json').write_text(json.dumps(r,indent=2)+'\n')
    summary=dict(fresh_checks=results,all_fresh_pass=all(r['passed'] for r in results),original_regression_usable=r['target'] is not None,
        independent_physical_validation=False,opencv=cv2.__version__)
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    files=[HERE/'vision.py',HERE/'pose.py',HERE/'inspection_policy.py',HERE/'check_perception.py',ROOT/'docs/autonomous_asset/perception_protocol.md']
    (out/'source').mkdir()
    for p in files:shutil.copyfile(p,out/'source'/p.name)
    (out/'freeze.json').write_text(json.dumps(dict(files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},checks=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    cv2.setNumThreads(1);main(Path(sys.argv[1]).resolve())
