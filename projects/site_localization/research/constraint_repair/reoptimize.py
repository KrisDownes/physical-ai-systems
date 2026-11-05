"""Offline SE(2) graph counterfactual. Edge removal is evaluator-selected ORACLE.

The objective uses recorded relative constraints and diagonal covariance, with
node zero fixed. Truth is read only after optimization for scoring.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'research'))
from continuous_asset_analysis import truth_pose


def rows(path):
    with path.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def solve(nodes, edges, drop, loss):
    ids = np.array([int(n['id']) for n in nodes])
    assert np.array_equal(ids, np.arange(len(ids)))
    a = np.array([int(e['source']) for e in edges if (int(e['source']), int(e['target'])) not in drop])
    b = np.array([int(e['target']) for e in edges if (int(e['source']), int(e['target'])) not in drop])
    es = [e for e in edges if (int(e['source']), int(e['target'])) not in drop]
    z = np.array([[e['dx'], e['dy'], e['dyaw']] for e in es])
    sig = np.sqrt(np.array([[e['var_x'], e['var_y'], e['var_yaw']] for e in es]))
    first = np.array([nodes[0][k] for k in ('x','y','yaw')])
    initial = np.array([[n[k] for k in ('x','y','yaw')] for n in nodes])
    def residual(flat):
        pose = np.vstack((first, flat.reshape(-1,3)))
        d = pose[b,:2] - pose[a,:2]
        c,s = np.cos(pose[a,2]), np.sin(pose[a,2])
        pred = np.stack((c*d[:,0]+s*d[:,1],-s*d[:,0]+c*d[:,1],wrap(pose[b,2]-pose[a,2])),axis=1)
        diff = (pred-z)/sig
        diff[:,2] = wrap(pred[:,2]-z[:,2])/sig[:,2]
        return diff.ravel()
    sparse = lil_matrix((3*len(es),3*(len(nodes)-1)),dtype=np.int8)
    for i,(u,v) in enumerate(zip(a,b)):
        if u:sparse[3*i:3*i+3,3*(u-1):3*u]=1
        if v:sparse[3*i:3*i+3,3*(v-1):3*v]=1
    fit=least_squares(residual,initial[1:].ravel(),jac='2-point',jac_sparsity=sparse.tocsr(),
                      method='trf',loss=loss,f_scale=.7,max_nfev=80,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    result=np.vstack((first,fit.x.reshape(-1,3)))
    return result,dict(cost=float(fit.cost),nfev=fit.nfev,success=bool(fit.success),message=fit.message,edges=len(es))


def main():
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--drop',action='append',default=[]);p.add_argument('--loss',default='linear',choices=['linear','huber','cauchy']);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    base=ROOT/'results/connected_layout_20261025';stem=base/'analysis'/f'{args.case}_graph'
    nodes=rows(Path(str(stem)+'_nodes.csv'));edges=rows(Path(str(stem)+'_edges.csv'))
    drops={tuple(map(int,x.split(':'))) for x in args.drop}
    poses,meta=solve(nodes,edges,drops,args.loss)
    truth=truth_pose(base/f'mission_{args.case}')
    times=np.array([n['time'] for n in nodes]);physical=np.stack([np.interp(times,truth[:,0],truth[:,i]) for i in (1,2)],axis=1)
    xy=poses[:,:2]-2.0;errors=np.linalg.norm(xy-physical,axis=1)
    saved=np.array([[n[k] for k in ('x','y')] for n in nodes])-2.0
    saved_error=np.linalg.norm(saved-physical,axis=1)
    meta.update(case=args.case,drop=sorted(map(list,drops)),loss=args.loss,
                node_314_error_m=float(errors[314]) if len(errors)>314 else None,
                endpoint_error_m=float(errors[-1]),median_error_m=float(np.median(errors)),
                max_error_m=float(errors.max()),saved_endpoint_error_m=float(saved_error[-1]),
                saved_median_error_m=float(np.median(saved_error)),saved_max_error_m=float(saved_error.max()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(meta,indent=2)+'\n')
    with args.out.with_suffix('.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['id','time','x','y','yaw','physical_error_m'])
        for i,n in enumerate(nodes):w.writerow([i,n['time'],*poses[i],errors[i]])
    print(json.dumps(meta))

if __name__=='__main__':main()
