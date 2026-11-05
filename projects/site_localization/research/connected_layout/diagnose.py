"""Read-only graph/pose audit of completed transfer attempts; no replay or tuning."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from continuous_asset_analysis import csv_pose,truth_pose

def load(p):return json.loads(p.read_text())
def csvrows(p):
    with p.open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
def at(t,truth):
    return np.array([np.interp(t,truth[:,0],truth[:,k]) for k in [1,2]]+[np.interp(t,truth[:,0],np.unwrap(truth[:,3]))])

def main(root):
    output=root/'analysis';diagnostics={}
    for case in load(root/'execution_freeze.json')['cases']:
        mission=root/('mission_'+case)
        if not (mission/'outcome.json').exists():continue
        truth=truth_pose(mission);tf=csv_pose(mission/'acquisition/mapper/mapper_tf_pose.csv');tf[:,1:3]-=2
        pose=csv_pose(mission/'acquisition/mapper/mapper_pose.csv');pose[:,1:3]-=2
        record={}
        for name,data in [('mapper_pose',pose),('mapper_tf',tf)]:
            errors=np.array([np.linalg.norm(row[1:3]-at(row[0],truth)[:2]) for row in data])
            indices=np.flatnonzero(errors>.3)
            if len(indices):
                j=int(indices[0]);record[name+'_first_error_over_0_3m']=dict(time=float(data[j,0]),estimate=data[j,1:].tolist(),
                    actual=at(data[j,0],truth).tolist(),error_m=float(errors[j]),previous_time=float(data[j-1,0]) if j else None,
                    previous_error_m=float(errors[j-1]) if j else None)
            jump=np.linalg.norm(np.diff(data[:,1:3],axis=0),axis=1);j=int(jump.argmax())
            record[name+'_largest_step']=dict(before=data[j].tolist(),after=data[j+1].tolist(),step_m=float(jump[j]),
                physical_step_m=float(np.linalg.norm(at(data[j+1,0],truth)[:2]-at(data[j,0],truth)[:2])))
        graph=mission/'acquisition/mapper/session.posegraph'
        if graph.exists():
            prefix=output/(case+'_graph')
            if not (output/(case+'_graph_nodes.csv')).exists():
                env=os.environ.copy();env['LD_LIBRARY_PATH']='/opt/ros/jazzy/lib:'+env.get('LD_LIBRARY_PATH','')
                reply=subprocess.run([str(output/'inspect_graph'),str(graph),str(prefix)],env=env,capture_output=True,text=True)
                (output/(case+'_graph_decode.log')).write_text(reply.stdout+reply.stderr)
                if reply.returncode:raise RuntimeError('Graph inspection failed: '+case)
            nodes={int(n['id']):n for n in csvrows(output/(case+'_graph_nodes.csv'))};edges=csvrows(output/(case+'_graph_edges.csv'))
            record['final_initial_graph_node']=nodes[0]
            bad=[]
            for edge in edges:
                source,target=int(edge['source']),int(edge['target']);a,b=at(nodes[source]['time'],truth),at(nodes[target]['time'],truth)
                c,s=np.cos(a[2]),np.sin(a[2]);d=b[:2]-a[:2];relative=np.array([c*d[0]+s*d[1],-s*d[0]+c*d[1]])
                error=float(np.linalg.norm(relative-[edge['dx'],edge['dy']]))
                if error>.5:
                    bad.append(dict(source=source,target=target,source_time=nodes[source]['time'],target_time=nodes[target]['time'],
                        measured_relative_xy=[edge['dx'],edge['dy']],physical_relative_xy=relative.tolist(),
                        translation_error_m=error,variance_xy=[edge['var_x'],edge['var_y']],
                        temporal_separation_s=nodes[target]['time']-nodes[source]['time']))
            record['graph_edge_errors_over_0_5m']=sorted(bad,key=lambda e:(e['target_time'],e['source_time']))
            record['graph_note']='Final serialized graph; edge birth branch/acceptance response not recorded. Error thresholds are descriptive diagnostics, never policy gates.'
        diagnostics[case]=record
    (output/'causal_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    if 'visible' in diagnostics:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        p=root/'mission_visible';truth=truth_pose(p);mapping=csv_pose(p/'acquisition/mapper/mapper_tf_pose.csv');mapping[:,1:3]-=2
        amcl=csv_pose(p/'inspection/estimates.csv');handoff=load(p/'handoff.json')['estimated_state']['time']
        fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
        ax[0].plot(truth[:,0],truth[:,1],label='physical x (judge)');ax[0].plot(mapping[:,0],mapping[:,1],label='SLAM x')
        ax[0].plot(amcl[:,0],amcl[:,1],label='AMCL x');ax[0].set_ylabel('Site x (m)');ax[0].legend()
        for label,data in [('SLAM',mapping),('AMCL',amcl)]:
            err=np.array([np.linalg.norm(r[1:3]-at(r[0],truth)[:2]) for r in data]);ax[1].plot(data[:,0],err,label=label)
        for a in ax:
            a.axvline(131.,color='red',ls='--',label='incorrect graph constraint at131s' if a is ax[1] else None)
            a.axvline(handoff,color='grey',ls=':',label='handoff' if a is ax[1] else None);a.grid(alpha=.2)
        ax[1].set(ylabel='Physical position error (m)',xlabel='Uninterrupted simulator time (s)');ax[1].legend()
        fig.suptitle('Visible control: map pose jumps before handoff; physical motion stays continuous')
        fig.tight_layout();fig.savefig(output/'visible_failure_trace.png',dpi=160);plt.close(fig)
    print({case:dict(first=record.get('mapper_pose_first_error_over_0_3m'),bad_graph_edges=len(record.get('graph_edge_errors_over_0_5m',[]))) for case,record in diagnostics.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
