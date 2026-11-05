"""Live-map frontier acquisition. No world, camera, asset or evaluator inputs."""
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research'))
sys.path.insert(0,str(ROOT/'research/moved_asset'))
sys.path.insert(0,str(ROOT/'research/autonomous_asset/reused_rover'))
from control import Mission,wrap
from safety import reserve
from buffered_planning import BufferedPlanner
from planning import GridPlanner
from rover_exploration.frontier_selection import find_frontier_cells,cluster_frontier_cells,build_reachable_candidates,select_weighted_goal
from rover_exploration.grid_planning import compute_reachable_component
import rclpy
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist


class Explorer(Mission):
    def __init__(self,out):
        super().__init__(out)
        self.done=True;self.phase='bootstrap';self.grid=None;self.map_seq=0;self.planned_seq=-1
        self.planner=None;self.last_plan=-10.;self.empty_since=None;self.leg_start=0.
        self.visited=[];self.failed=[];self.goal_xy=None;self.goal_count=0;self.wheel_distance=0.;self.previous_wheel=None
        self.termination=None;self.finished=False;self.last_progress=0.;self.progress_pose=None
        self.spin_angle=0.;self.spin_previous=None;self.scan_min=0.;self.bootstrap_end=None
        self.cfg.pop('dock_completion',None)
        self.require_return=json.loads((out/'exploration_config.json').read_text())['require_return']
        self.handoff_stationary_since=None;self.handoff_written=False;self.last_covariance=None
        self.decisions=(out/'exploration.jsonl').open('w');(out/'decision_maps').mkdir()
        qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid,'/map',self.on_map,qos)

    def on_map(self,m):
        q=m.info.origin.orientation
        if abs(q.z)>1e-6 or abs(q.x)>1e-6 or abs(q.y)>1e-6:
            self.finish('unsupported_rotated_map');return
        self.grid=m;self.map_seq+=1

    def record(self,kind,**fields):
        row=dict(type=kind,time=self.sim_time,pose=None if self.pose is None else self.pose.tolist())
        row.update(fields);self.decisions.write(json.dumps(row)+'\n');self.decisions.flush()

    def on_scan(self,m):
        super().on_scan(m)
        ranges=np.asarray(m.ranges)
        self.scan_min=float(np.min(ranges)) if len(ranges) and not np.isnan(ranges).any() else 0.

    def finish(self,reason):
        if self.finished:return
        self.finished=True;self.done=True;self.phase='finished';self.cmd.publish(Twist())
        record=dict(reason=reason,exploration_termination=self.termination,time=self.sim_time,
                    wheel_distance_m=self.wheel_distance,frontier_goals=self.goal_count,
                    failed_goals=self.failed,visited=self.visited,map_messages=self.map_seq,
                    known_free_area_m2=None if self.planner is None else float(np.sum(np.asarray(self.grid.data)==0)*self.grid.info.resolution**2),
                    complete_environment_claim=False)
        (self.out/'acquisition_done.json').write_text(json.dumps(record,indent=2)+'\n')
        self.record('finished',**record)

    def refresh(self):
        m=self.grid;i=m.info;raw=np.asarray(m.data,dtype=int).reshape(i.height,i.width)
        native=np.zeros_like(raw);native[raw==0]=-1;native[raw==100]=1
        path=self.out/'live_planning_grid.txt'
        # ROS origin is the corner; GridPlanner takes the center of cell(0,0).
        with path.open('w') as f:
            f.write(f'{i.width} {i.height} {i.resolution} {i.origin.position.x+i.resolution/2} {i.origin.position.y+i.resolution/2}\n')
            np.savetxt(f,native,fmt='%d')
        self.planner=BufferedPlanner(path);self.safety_grid=self.planner.hard;self.planned_seq=self.map_seq;self.last_plan=self.sim_time
        return raw.ravel().tolist()

    def install(self,path,goal,yaw,phase):
        points=path['points'][1:]
        if not points:points=[goal]
        poses=[]
        for n,p in enumerate(points):
            heading=yaw if n==len(points)-1 else math.atan2(points[n+1][1]-p[1],points[n+1][0]-p[0])
            poses.append([*p,heading])
        self.route=[dict(pose=p,inspection_index=None) for p in poses];self.targets=np.asarray(poses)
        self.route_step=0;self.done=False;self.dwell=None;self.phase=phase;self.goal_xy=list(goal)
        self.leg_start=self.sim_time;self.last_progress=self.sim_time;self.progress_pose=self.pose[:2].copy()
        self.record('path',phase=phase,goal=goal,points=path['points'])

    def return_home(self,reason):
        self.termination=reason;self.record('return_requested' if self.require_return else 'handoff_requested',reason=reason)
        if not self.require_return:
            self.finish('mapping_endpoint_ready');return
        path=self.planner.path(self.pose,[0.,0.])
        if path is None:self.finish('dock_path_unavailable');return
        self.cfg['dock_completion']=dict(stationary_dwell_s=2)
        self.install(path,[0.,0.],0.,'return')

    def choose(self,raw):
        p=self.planner;start=p.start_index(self.pose)
        if start is None:self.return_home('robot_outside_planning_free_space');return
        planning=np.where(p.free,0,100).ravel().tolist()
        bfs=compute_reachable_component(planning,p.w,p.h,(start//p.w,start%p.w))
        frontier=find_frontier_cells(raw,p.w,p.h);clusters=cluster_frontier_cells(frontier,5)
        candidates=build_reachable_candidates(raw,planning,p.w,p.h,clusters,bfs,int(math.ceil(1.5/p.res)))
        area=float(sum(v==0 for v in raw)*p.res**2);eligible=[]
        for cell in candidates.sizes:
            xy=p.point(cell[0]*p.w+cell[1])
            if np.linalg.norm(xy-self.pose[:2])<.3:continue
            if any(np.linalg.norm(xy-np.array(v['point']))<.4 and area-v['area']<.5 for v in self.visited):continue
            nearby=[v for v in self.failed if np.linalg.norm(xy-np.array(v['point']))<.5]
            if len(nearby)>=2 or any(self.sim_time-v['time']<20 for v in nearby):continue
            eligible.append(cell)
        selected=select_weighted_goal(bfs,candidates,eligible,int(.5/p.res)) if eligible else None
        number=len(list((self.out/'decision_maps').glob('*.txt')))
        snapshot=self.out/'decision_maps'/f'{number:03d}.txt';snapshot.write_bytes((self.out/'live_planning_grid.txt').read_bytes())
        self.record('frontier_decision',map_file=str(snapshot),map_stamp=self.grid.header.stamp.sec+self.grid.header.stamp.nanosec*1e-9,
            free_area_m2=area,raw_frontiers=len(frontier),clusters=len(clusters),unreachable_clusters=candidates.unreachable_clusters,
            candidates=[dict(cell=list(c),point=p.point(c[0]*p.w+c[1]).tolist(),size=n,eligible=c in eligible) for c,n in candidates.sizes.items()],
            selected=None if selected is None else list(selected[0]),selection='shortest reachable BFS distance plus 0.5m slack; prefer larger cluster')
        if selected is None:
            if self.empty_since is None:self.empty_since=self.sim_time
            if self.sim_time-self.empty_since>=8:self.return_home('no_eligible_frontier')
            return
        self.empty_since=None;cell,_=selected;goal=p.point(cell[0]*p.w+cell[1]);path=p.path(self.pose,goal)
        if path is None:self.record('planning_disagreement',goal=goal.tolist());self.finish('planning_disagreement');return
        if self.require_return:
            home=p.path(goal,[0,0])
            decision=reserve(self.wheel_distance,path['length_m'],home['length_m'] if home else float('inf'),self.sim_time,len(path['points'])-1)
            self.record('return_reserve_check',goal=goal.tolist(),**decision)
            if not decision['admit']:self.return_home('return_reserve');return
        if not self.require_return and (self.wheel_distance+1.25*path['length_m']+.5>15 or self.sim_time+1.25*path['length_m']/.18+8*(len(path['points'])-1)+15>180):
            self.return_home('continuous_acquisition_reserve');return
        anchor=candidates.anchors[cell];look=p.point(anchor[0]*p.w+anchor[1]);heading=math.atan2(look[1]-goal[1],look[0]-goal[0])
        self.goal_count+=1;self.install(path,goal.tolist(),heading,'frontier')

    def tick(self):
        super().tick()
        if self.finished:
            self.prepare_handoff();return
        if self.odom:
            xy=self.odom[-1][1][:2]
            if self.previous_wheel is not None:self.wheel_distance+=float(np.linalg.norm(xy-self.previous_wheel))
            self.previous_wheel=xy.copy()
        if self.sim_time>=240 or self.wheel_distance>=20:self.finish('total_acquisition_budget');return
        if not self.require_return and (self.sim_time>=180 or self.wheel_distance>=15):
            self.return_home('continuous_acquisition_budget');return
        if self.grid is None or self.pose is None or self.slam_pose is None:return
        if time.monotonic()-self.tf_wall>=1 or not(-.1<=self.sim_time-self.tf_stamp<1):return
        if self.sim_time<2:return
        if self.phase=='bootstrap':
            # A first sparse scan cannot satisfy min_pass_through=2 far from the
            # sensor. Gather actual shifted rays by one bounded in-place turn;
            # never fill unknown cells or use a hidden initial survey segment.
            heading=self.odom[-1][1][2]
            if self.spin_previous is not None:self.spin_angle+=abs(wrap(heading-self.spin_previous))
            self.spin_previous=heading
            if self.scan_min<=.4 or time.monotonic()-self.last_scan_wall>=1:
                self.finish('bootstrap_rotation_obstructed');return
            if self.spin_angle<2*math.pi and self.sim_time<30:
                cmd=Twist();cmd.angular.z=.35;self.cmd.publish(cmd);return
            self.cmd.publish(Twist())
            if self.spin_angle<2*math.pi:self.finish('bootstrap_rotation_timeout');return
            if self.bootstrap_end is None:
                self.bootstrap_end=self.sim_time;self.record('bootstrap_rotation_complete',wheel_angle_rad=self.spin_angle)
            if self.sim_time-self.bootstrap_end<2:return
            self.phase='idle'
        raw=None
        if self.planner is None or (self.map_seq!=self.planned_seq and self.sim_time-self.last_plan>=2):raw=self.refresh()
        if self.require_return and self.phase=='frontier' and raw is not None:
            home=self.planner.path(self.pose,[0,0])
            if home is not None:
                check=reserve(self.wheel_distance,0,home['length_m'],self.sim_time,1)
                if not check['admit']:
                    self.done=True;self.cmd.publish(Twist());self.return_home('return_reserve_during_leg');return
        if self.phase in ['frontier','return']:
            if self.done:
                self.record('goal_reached',phase=self.phase,goal=self.goal_xy)
                if self.phase=='return':self.finish('returned_to_dock');return
                self.visited.append(dict(point=self.goal_xy,area=float(np.sum(np.asarray(self.grid.data)==0)*self.grid.info.resolution**2)))
                self.phase='idle'
            else:
                if np.linalg.norm(self.pose[:2]-self.progress_pose)>=.08:
                    self.last_progress=self.sim_time;self.progress_pose=self.pose[:2].copy()
                blocked=raw is not None and not self.safety_grid.segment_free(self.pose[:2],self.targets[self.route_step,:2])
                failed=self.sim_time-self.leg_start>60 or self.sim_time-self.last_progress>20
                if blocked or failed:
                    self.cmd.publish(Twist());self.done=True
                    self.record('goal_interrupted',blocked_by_new_map=blocked,progress_timeout=failed,goal=self.goal_xy)
                    if self.phase=='return':self.finish('dock_navigation_failure');return
                    self.failed.append(dict(point=self.goal_xy,time=self.sim_time,blocked=blocked))
                    self.phase='idle'
                else:return
        if self.phase=='idle':
            if raw is None and self.sim_time-self.last_plan<2:return
            if raw is None:raw=self.refresh()
            if self.sim_time>=180 or self.wheel_distance>=15 or self.goal_count>=12 or len(self.failed)>=3:
                self.return_home('exploration_budget_or_failure_limit');return
            self.choose(raw)

    def on_slam(self,m):
        super().on_slam(m)
        self.last_covariance=list(m.pose.covariance)

    def prepare_handoff(self):
        if self.handoff_written or self.pose is None or len(self.odom)<2:return
        t,p=self.odom[-1];last,q=self.odom[-2];dt=t-last
        stationary=dt>0 and np.linalg.norm(p[:2]-q[:2])/dt<.02 and abs(wrap(p[2]-q[2]))/dt<.04
        fresh=-.1<=self.sim_time-self.tf_stamp<.5
        if not stationary or not fresh:self.handoff_stationary_since=None;return
        if self.handoff_stationary_since is None:self.handoff_stationary_since=self.sim_time
        if self.sim_time-self.handoff_stationary_since<2:return
        if self.last_covariance is None:return
        value=dict(time=self.sim_time,pose_time=self.estimate_time,map_pose=self.pose.tolist(),wheel_time=t,wheel_pose=p.tolist(),slam_covariance=self.last_covariance,stationary_since=self.handoff_stationary_since,source='SLAM exact-time TF and unreset wheel odometry',truth_used=False)
        (self.out/'handoff_ready.json').write_text(json.dumps(value,indent=2)+'\n')
        self.handoff_written=True;self.record('handoff_ready',**value)

    def close(self):
        self.decisions.close();super().close()


if __name__=='__main__':
    rclpy.init();node=Explorer(Path(sys.argv[1]))
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.close();node.destroy_node();rclpy.try_shutdown()
