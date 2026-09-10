"""Offline diagnosis of a saved onboard observation; never commands a rover."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rover_exploration.frontier_node import FrontierDetector
from rover_exploration.grid_planning import compute_reachable_component, reconstruct_grid_path, inflate_occupancy_grid, build_planning_grid
from rover_exploration.frontier_selection import find_frontier_cells, cluster_frontier_cells, representative_frontier_cell, find_reachable_approach, world_point_to_grid_cell, grid_cell_center


def diagnose(source,output):
    output.mkdir(parents=True,exist_ok=True);(output/'COLCON_IGNORE').touch()
    payloads=[json.loads(l) for l in (source/'observations.jsonl').read_text().splitlines()]
    observations=[p['structuredContent'] for p in payloads];last=observations[-1]
    target=last['slam_map']['captured_sim_time_s'];mapping=None;odom=[]
    reader=SequentialReader();reader.open(StorageOptions(uri=str(source/'bag'),storage_id='mcap'),ConverterOptions('cdr','cdr'))
    types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in ('/map','/odometry/filtered')}
    while reader.has_next():
        topic,data,_=reader.read_next()
        if topic not in types:continue
        msg=deserialize_message(data,types[topic]);stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if topic=='/map' and abs(stamp-target)<1e-6:mapping=msg
        elif topic=='/odometry/filtered':odom.append((stamp,msg.pose.pose.position.x,msg.pose.pose.position.y))
    if mapping is None:raise RuntimeError('exact model-visible map timestamp missing from bag')
    info=mapping.info;w,h,res=info.width,info.height,info.resolution
    pose=last['estimated_pose'];start=world_point_to_grid_cell(pose['x_m'],pose['y_m'],res,info.origin.position.x,info.origin.position.y)
    config=SimpleNamespace(rover_length_m=.45,rover_width_m=.30,path_clearance_m=.05,wall_closing_radius_m=.05,unknown_clearance_m=.1)
    native,inflation=FrontierDetector._build_planning_grid(config,mapping)
    clusters=cluster_frontier_cells(find_frontier_cells(mapping.data,w,h),5)
    analyses=[]
    for name,grid in [('native',native),('clearance_0_45m',build_planning_grid(mapping.data,inflate_occupancy_grid(mapping.data,w,h,math.ceil(.45/res)),native))]:
        bfs=compute_reachable_component(grid,w,h,start);candidates=[]
        for i,cluster in enumerate(clusters):
            representative=representative_frontier_cell(cluster)
            approach=find_reachable_approach(mapping.data,grid,w,h,cluster,bfs,math.ceil(1.5/res))
            point=lambda cell:list(grid_cell_center(*cell,res,info.origin.position.x,info.origin.position.y))
            candidates.append(dict(id=f'F{i+1}',cells=len(cluster),representative_xy_m=point(representative),
                approach_xy_m=point(approach) if approach else None,
                path_length_grid_m=bfs['cost'][approach]*res if approach else None,
                path_cells=reconstruct_grid_path(bfs['came_from'],approach) if approach else None))
        analyses.append(dict(configuration=name,start_traversable=bfs is not None,frontiers=candidates))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    raster=np.array(mapping.data).reshape(h,w)
    raster=np.where(raster<0,0,np.where(raster==0,1,2))
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for axis,analysis in zip(axes,analyses):
        axis.imshow(raster,origin='lower',cmap=ListedColormap(['.6','white','black']),vmin=0,vmax=2,
                    extent=[info.origin.position.x,info.origin.position.x+w*res,info.origin.position.y,info.origin.position.y+h*res])
        for c in analysis['frontiers']:
            if c['path_cells']:
                points=[grid_cell_center(*cell,res,info.origin.position.x,info.origin.position.y) for cell in c['path_cells']]
                line,=axis.plot([p[0] for p in points],[p[1] for p in points],label=c['id'],linewidth=1.3)
                axis.scatter(*c['approach_xy_m'],marker='*',color=line.get_color(),s=65)
            axis.text(*c['representative_xy_m'],c['id'],color='red',fontsize=9)
        axis.scatter(pose['x_m'],pose['y_m'],color='blue',label='saved pose',s=25)
        axis.set(title=analysis['configuration'],xlabel='Online map x (m)',ylabel='Online map y (m)')
        axis.legend(fontsize=7)
    fig.suptitle('Offline known-free routes to approach points: not physical reachability proof')
    fig.tight_layout();fig.savefig(output/'frontier-approaches.png',dpi=150);plt.close(fig)
    events=[json.loads(l) for l in (source/'events.jsonl').read_text().splitlines()]
    timing=json.loads((source/'action_timeline.json').read_text());physical=json.loads((source/'physical_actions.json').read_text())
    aborted=[]
    for i,obs in enumerate(observations):
        a=obs.get('last_completed_action_result')
        if not a or a['terminal_state']!='aborted':continue
        before=min(odom,key=lambda p:abs(p[0]-a['accepted_sim_time_s']))
        after=max((p for p in odom if p[0]<=a['terminal_sim_time_s']),key=lambda p:p[0])
        dist=math.hypot(after[1]-before[1],after[2]-before[2])
        row=next(t for t in timing if t['id']==a['command_id'])
        accepted=next(e for e in row['events'] if e['kind']=='executor_accepted')
        terminal=next(e for e in row['events'] if e['kind']=='executor_status' and e['state']=='aborted')
        stops=[e for e in events if e['kind']=='command' and e['data']['action']=='stop' and accepted['wall_s']<=e['wall_s']<=terminal['wall_s']]
        aborted.append(dict(observation_index=i,result=a,execution_wall_s=row['execution_s'],
            measured_action_rtf=a['execution_sim_time_s']/row['execution_s'],filtered_travel_m=dist,remaining_m=a['distance_m']-dist,
            stop_commands_during_action=stops,before=observations[i-1],after=obs,
            physical=next(p for p in physical if p['command']['id']==a['command_id'])))
    selected=sorted(set(range(len(payloads)-7,len(payloads)))|{j for a in aborted for j in (a['observation_index']-1,a['observation_index'])})
    for i in selected:
        for k,c in enumerate(c for c in payloads[i]['content'] if c['type']=='image'):
            (output/f'observation-{i:02}-{k}.png').write_bytes(base64.b64decode(c['data']))
    audit=json.loads((source/'driver_audit.json').read_text())
    result=dict(source=str(source),map_sim_s=target,pose_sim_s=pose['captured_sim_time_s'],map_origin=dict(x=info.origin.position.x,y=info.origin.position.y),
        map_dimensions=[w,h],resolution_m=res,native_inflation_cells=inflation,approach_search_radius_m=1.5,
        frontier_analysis=analyses,aborted_drives=aborted,last_observations=observations[-7:],
        finish_calls=[c for c in audit['calls'] if c.get('tool')=='finish'],messages=audit['messages'],
        limitation='Offline map-connectivity/vantage analysis only. Unknown interiors, map errors, guard geometry and physical traversal are not certified; approach reachability does not establish complete frontier reachability.')
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2))
    files=['driver.log','observations.jsonl','manifest.json','results.json','bag/metadata.yaml']
    (output/'original_evidence_hashes.json').write_text(json.dumps({f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},indent=2))
    print(json.dumps({**{k:v for k,v in result.items() if k in ('map_sim_s','map_dimensions','native_inflation_cells')},'frontiers':[{**a,'frontiers':[{k:v for k,v in c.items() if k!='path_cells'} for c in a['frontiers']]} for a in analyses], 'aborted':[{'id':a['result']['command_id'],**{k:a[k] for k in ('execution_wall_s','measured_action_rtf','filtered_travel_m','remaining_m','stop_commands_during_action')}} for a in aborted]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();diagnose(a.source.resolve(),a.output.resolve())
