"""One paired pilot. Reuses process ownership, sensor monitor and original evaluator."""
import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
import yaml
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from episode_cancellation import latch_cancel
import rover_experiment as infrastructure
from rover_experiment import ROOT, Monitor, Processes, wait_for, save, audit, model_metadata
from exploration_measurement import summarize, announce_completion, benchmark_bag
from verify_rover_bags import verify

ACTIVE_CANCEL_PATH=None
INTERRUPTED=False
CLEANING_UP=False

TOPICS=list(dict.fromkeys(infrastructure.TOPICS+['/planned_path','/cmd_vel_raw','/recovery_request','/guard_intervention']))
TASK='''Explore this unknown environment using only rover observe/drive/turn/stop/finish.
Observe once initially. Every movement response includes a fresh observation; use observe again only for recovery.
Use camera, lidar, online SLAM map, estimated pose and unranked frontier candidates to choose EVERY movement yourself.
Choose useful frontiers and routes, remembering visited places and failed approaches. Candidates are not guaranteed reachable.
Drive bounds +/-0.5 meters, turn bounds +/-90 degrees; local obstacle protection cannot be disabled.
Respond with concise tool arguments only, no repeated scene descriptions. Execute one action at a time.
The compact action ledger retains all prior movement results. Use it and prior poses/images to avoid repeating failures.
After an obstacle abort choose a different safe action, never a scripted fallback.
Continue until supervisor stops you, 100 movements, or no safe continuation; then finish with a short reason.
No shell, files, outside tools, ground truth or evaluator information. Do not claim success from action status alone.
A wall or blocked movement establishes only local blockage. Before declaring exploration blocked, inspect the online map and remaining frontier candidates for an alternate approach or a route back through previously observed free space. If a plausible safe continuation exists, choose and execute it, then reassess using fresh observations. Before finish(blocked), provide a concise explanation identifying the remaining candidates considered and the observed reason continuation is unsafe or unavailable. Treat uncertain reachability as uncertain. Emergency stopping remains available at all times.'''


def classify(termination, log, passed, evidence):
    # Positive service evidence only, never SIGTERM inference.
    quota=('usage limit', 'insufficient_quota', 'quota exceeded', 'quota_exhausted')
    errors=[]
    for line in log.splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        if event.get('type') in ('error','turn.failed'):
            errors.append(json.dumps(event).lower())
    if any(s in e for e in errors for s in quota): return 'quota_exhaustion'
    if termination in ('interrupted','infrastructure_error','timeout','action_limit','quota_exhaustion','decision_turn_limit'): return termination
    return 'success' if passed and evidence else 'navigation_failure'


def controller_ready(graph, method, hierarchical=False):
    expected=['agent_executor','experiment_supervisor'] if method=='llm' else ['experiment_supervisor','obstacle_guard']
    if hierarchical: expected=['experiment_supervisor','frontier_detector']
    commands=graph.get('command_publishers',[])
    expected_commands=['experiment_supervisor']
    if method=='llm' and 'rover_driver_interface' in commands:
        expected_commands.append('rover_driver_interface')
    return (sorted(graph['velocity_publishers'])==expected
            and (not hierarchical or graph.get('guarded_velocity_publishers')==['obstacle_guard'])
            and sorted(commands)==sorted(expected_commands)
            and sorted(graph.get('raw_velocity_publishers',[]))==(['path_follower'] if method=='classical' else []))


def selected_methods(method, resume_unstarted=False):
    if resume_unstarted and method != 'both':
        raise ValueError('--resume-unstarted cannot be combined with --method')
    return ('llm',) if resume_unstarted else ('classical','llm') if method=='both' else (method,)


def cleanup_owned_episode(processes, cancel_path, publications):
    """Latch usage cancellation first; ROS failures cannot bypass owned teardown."""
    errors = []
    latch_cancel(cancel_path)
    for publish in publications:
        try:publish()
        except Exception as error:errors.append('stop_publication: '+str(error))
    # Stop the selector/controller first, recorder last. Never enumerate/kill peers.
    entries = list(reversed(processes.owned))
    entries.sort(key=lambda e: 0 if e[0] in ('driver','frontier_detector') else 2 if e[0]=='recorder' else 1)
    for entry in entries:
        try:
            processes.stop(entry)
            processes.owned.remove(entry)
        except Exception as error:errors.append(entry[0]+': '+str(error))
    save(processes.folder/'cleanup.json',processes.cleanup)
    return errors


def run_episode(folder,method,args):
    global ACTIVE_CANCEL_PATH, CLEANING_UP
    CLEANING_UP=False
    hierarchy=getattr(args,'hierarchical_selector',None)
    budget=getattr(args,'wall_budget',600)
    folder.mkdir(); processes=Processes(folder); monitor=None; executor=None; spin=None
    cancel=None; emergency=None
    cancel_path=folder/"cancel.requested"
    ACTIVE_CANCEL_PATH=cancel_path
    driver=None; started=None; ended=None; manifest=dict(method=method,model_requested=args.model if method=='llm' else None,
        reasoning_effort='low' if method=='llm' else None,ros_domain_id=87,wall_budget_s=budget,
        action_budget=100 if method=='llm' else None,world=args.world,simulator_seed=args.seed,spawn=dict(x=0,y=0,z=.02,yaw=0),
        setup_started_monotonic_s=time.monotonic(),termination='infrastructure_error')
    manifest.update(benchmark_mode='hierarchical' if hierarchy else ('direct_llm' if method=='llm' else 'original_classical'), selector=hierarchy, diagnostic=hierarchy=='mock')
    if hierarchy=='llm': manifest.update(model_requested='gpt-5.6-luna',reasoning_effort='low',decision_turn_budget=20)
    empty=tempfile.TemporaryDirectory(prefix='rover-pilot-driver-')
    env=dict(os.environ,GZ_PARTITION='rover-pilot-'+folder.parent.name+'-'+method,
             ROVER_CANCEL_FILE=str(cancel_path.resolve()),ROVER_OBSERVATION_MODE='onboard',ROVER_OBSERVATION_LOG=str(folder/'observations.jsonl'))
    if hierarchy: env.update(ROVER_SELECTOR=hierarchy,ROVER_HIERARCHY_LOG=str(folder),ROVER_ENABLE_MODEL='1' if getattr(args,'enable_model',False) else '0')
    try:
        probe=Node('experiment_preflight')
        for _ in range(20):rclpy.spin_once(probe,timeout_sec=.1)
        existing=[n for n in probe.get_node_names() if n!='experiment_preflight'];probe.destroy_node()
        if existing:raise RuntimeError('ROS domain occupied: '+str(existing))
        monitor=Monitor(folder,publish_completion=method=='llm')
        monitor.native_complete=False;monitor.native_result=None
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        monitor.create_subscription(Bool,'/exploration_complete',lambda m:setattr(monitor,'native_complete',m.data),qos)
        monitor.create_subscription(String,'/exploration_result',lambda m:setattr(monitor,'native_result',json.loads(m.data)),qos)
        monitor.create_subscription(Bool,'/guard_intervention',lambda m:monitor.event('obstacle',m.data),10)
        monitor.create_subscription(Twist,'/cmd_vel_raw',lambda m:monitor.event('raw_velocity',[m.linear.x,m.angular.z]),10)
        emergency=monitor.create_publisher(Twist,'/cmd_vel',10)
        cancel=monitor.create_publisher(String,'/hierarchy_cancel',10) if hierarchy else None
        monitor.hierarchy_status={}
        if hierarchy: monitor.create_subscription(String,'/hierarchy_status',lambda m:setattr(monitor,'hierarchy_status',json.loads(m.data)),10)
        executor=SingleThreadedExecutor();executor.add_node(monitor)
        spin=threading.Thread(target=executor.spin,daemon=True);spin.start()
        recorder=processes.start('recorder',['ros2','bag','record','-o',str(folder/'bag'),*TOPICS,*(['/hierarchy_status','/hierarchy_cancel','/cmd_vel_guarded','/recovery_status'] if hierarchy else [])],env=env)
        wait_for(lambda:'Listening for topics' in (folder/'recorder.log').read_text(),20,'recorder',[recorder])
        world=ROOT/'src/rover_description/worlds'/f'{args.world}.sdf'
        simulation_cmd=['ros2','launch','rover_exploration','exploration.launch.py',
             'agent_mode:='+str(method=='llm').lower(),'enable_motion:=false','start_frontier:=false',
             'agent_stop_distance:=0.45','enable_rviz:=false',
             'gazebo_args:=-r -s --headless-rendering'+(' --seed '+str(args.seed) if args.seed is not None else ''),
             'spawn_x:=0','spawn_y:=0','spawn_z:=0.02','spawn_yaw:=0',
             'world_name:='+args.world,'world_file:='+str(world)]
        manifest['simulation_command']=simulation_cmd
        simulation=processes.start('simulation',simulation_cmd,env=env)
        if not args.headless and os.environ.get('DISPLAY'):
            processes.start('gazebo_gui',['gz','sim','-g'],env=env)
            manifest['gui_requested']=True
        else:manifest['gui_requested']=False
        wait_for(lambda:all(k in monitor.latest and time.monotonic()-monitor.latest[k][1]<2
                            for k in ('camera','scan','odom','truth')) and monitor.coverage is not None,
                 90,'sensors and online map',[simulation,recorder])
        # Both methods begin with fresh stationary SLAM. Only now enable control.
        started=time.monotonic();start_sim=monitor.latest['truth'][0]
        manifest.update(episode_started_monotonic_s=started,episode_started_sim_s=start_sim)
        monitor.event('episode_start',dict(method=method,sim_s=start_sim))
        if method=='classical':
            guard=['ros2','run','rover_control','obstacle_guard','--ros-args','-p','use_sim_time:=true',
                   '-p','rear_stop_distance:=0.45','-p','rear_resume_distance:=0.55']
            follower=['ros2','run','rover_control','path_follower','--ros-args','-p','use_sim_time:=true']
            frontier=['ros2','run','rover_exploration','frontier_detector','--ros-args','-p','use_sim_time:=true']
            if hierarchy:
                guard += ['-r','/cmd_vel:=/cmd_vel_guarded']
                frontier=[str(ROOT/'.driver_venv/bin/python'),str(ROOT/'scripts/hierarchical_controller.py'),'--ros-args','-p','use_sim_time:=true']
            manifest['controller_commands']=[guard,follower,frontier]
            processes.start('frontier_detector',frontier,env=env)
            processes.start('obstacle_guard',guard,env=env);processes.start('path_follower',follower,env=env)
        else:
            infrastructure.TASK=TASK
            cmd,config=infrastructure.driver_command(args.model,Path(empty.name))
            extra={'ROVER_OBSERVATION_MODE':'onboard','ROVER_OBSERVATION_LOG':str(folder/'observations.jsonl'),'ROVER_MOVEMENT_LIMIT':'100'}
            for k,v in extra.items():
                key='mcp_servers.rover.env.'+k;config[key]=v;cmd[-1:-1]=['-c',key+'='+json.dumps(v)]
            save(folder/'driver_config.json',config);save(folder/'driver_command.json',cmd)
            driver=processes.start('driver',cmd,env=env)
        wait_for(lambda:controller_ready(monitor.graph(),method,bool(hierarchy)),15,
                 'resolved exclusive controller publisher identities',[simulation,recorder])
        manifest['graph']=monitor.graph()
        save(folder/'manifest.json',manifest)
        last_print=0
        while True:
            now=time.monotonic();elapsed=now-started
            if hierarchy and monitor.hierarchy_status.get('terminal') and not monitor.native_complete:
                manifest['termination']=monitor.hierarchy_status.get('reason','navigation_failure');break
            if getattr(args,'stop_after_arrival',False) and monitor.hierarchy_status.get('arrivals',0)>=1:
                manifest['termination']='diagnostic_arrival';break
            if elapsed>=budget:manifest['termination']='timeout';break
            complete=monitor.native_complete if method=='classical' else monitor.completed
            if complete:
                manifest['termination']='completion';manifest['completion_wall_s']=elapsed;break
            if method=='classical' and monitor.native_result and monitor.native_result.get('outcome')=='blocked':
                manifest['termination']='navigation_failure';break
            if driver and driver.poll() is not None:
                manifest['termination']='driver_exit';break
            movements=[c for c in monitor.commands if c['action'] in ('drive','turn')]
            if method=='llm' and len(movements)>=100:
                last=movements[-1]['id']
                if any(s['id']==last and s['state'] in ('succeeded','rejected','aborted') for s in monitor.statuses):
                    manifest['termination']='action_limit';break
            for name,p,_ in processes.owned:
                if name not in ('driver','gazebo_gui') and p.poll() is not None:
                    raise RuntimeError(name+' exited '+str(p.returncode))
            if elapsed-last_print>=15:
                print(method,round(elapsed,1),'s',monitor.coverage,flush=True);last_print=elapsed
                save(folder/'live.json',dict(wall_s=elapsed,coverage=monitor.coverage,movements=len(movements)))
            time.sleep(.05)
        ended=time.monotonic();manifest['episode_ended_sim_s']=monitor.latest['truth'][0]
        latch_cancel(cancel_path)
        if cancel:
            cancel.publish(String(data=manifest['termination']));emergency.publish(Twist());time.sleep(.15)
        # Terminate owned command producers before emergency zero / completion.
        for entry in list(reversed(processes.owned)):
            if entry[0] in ('driver','path_follower','obstacle_guard','frontier_detector'):
                processes.stop(entry);processes.owned.remove(entry)
        monitor.stop_pub.publish(String(data=json.dumps(dict(id='pilot-stop',action='stop'))))
        for _ in range(5):emergency.publish(Twist());time.sleep(.05)
        if manifest['termination']=='diagnostic_arrival':
            end_sim=monitor.latest['truth'][0]+2
            wait_for(lambda:monitor.latest['truth'][0]>=end_sim,10,'diagnostic post-stop recording',[simulation])
        if manifest['termination']=='completion':
            if method=='llm':announce_completion(monitor)
            end_sim=monitor.latest['truth'][0]+5
            wait_for(lambda:monitor.latest['truth'][0]>=end_sim,30,'post completion recording',[simulation])
    except (Exception,KeyboardInterrupt) as error:
        manifest['error']=str(error)
        manifest['termination']='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'infrastructure_error'
    finally:
        CLEANING_UP=True
        latch_cancel(cancel_path)
        if started:
            ended=ended or time.monotonic();manifest['episode_wall_s']=ended-started
            if monitor.latest.get('truth'):
                manifest.setdefault('episode_ended_sim_s',monitor.latest['truth'][0])
                manifest['episode_sim_s']=manifest['episode_ended_sim_s']-manifest['episode_started_sim_s']
                manifest['real_time_factor']=manifest['episode_sim_s']/manifest['episode_wall_s']
        publications=[]
        if cancel is not None:publications.append(lambda:cancel.publish(String(data='shutdown')))
        if emergency is not None:publications.append(lambda:emergency.publish(Twist()))
        if monitor is not None:publications.append(lambda:monitor.stop_pub.publish(String(data=json.dumps(dict(id='pilot-cleanup',action='stop')))))
        manifest['cleanup_errors']=cleanup_owned_episode(processes,cancel_path,publications)
        if executor:
            try:executor.shutdown();spin.join(2);monitor.destroy_node()
            except Exception as error:manifest['cleanup_errors'].append('ROS teardown: '+str(error))
            finally:monitor.stream.close()
        empty.cleanup()
        manifest['cleanup_complete']=not processes.owned and all(p['group_gone'] for p in processes.cleanup)
        manifest['driver_exit_code']=driver.returncode if driver else None
        manifest['ended_monotonic_s']=time.monotonic()
        save(folder/'manifest.json',manifest)
        try:evaluate(folder,manifest)
        except Exception as error:
            save(folder/'results.json',dict(outcome='infrastructure_error',evaluation_error=str(error),manifest=manifest))
        print('Finished',method,manifest['termination'],folder,flush=True)
    return manifest


def session_evidence(folder):
    """Read only this driver's rollout; preserve exposed usage on interrupted turns."""
    metadata=model_metadata(folder)
    records=[];total=None;compactions=[]
    thread=metadata.get('thread_id')
    if thread:
        for path in (Path.home()/'.codex/sessions').rglob('*'+thread+'*.jsonl'):
            for line in path.read_text().splitlines():
                try:event=json.loads(line)
                except ValueError:continue
                payload=event.get('payload',{})
                if event.get('type')=='token_usage_record':
                    records.append(event)
                if event.get('type')=='event_msg' and payload.get('type')=='token_count' and payload.get('info'):
                    total=payload['info'].get('total_token_usage')
                if event.get('type')=='compacted':
                    compactions.append({'timestamp':event.get('timestamp'),'ordinal':event.get('ordinal')})
    value=dict(metadata=metadata,usage_records=records,total_token_usage=total,
               compaction_events=compactions,model_request_monotonic_s=None,
               model_response_monotonic_s=None,
               timing_note='Usage records have UTC response timestamps and response IDs; no monotonic inference request/response pair is exposed.')
    save(folder/'session_evidence.json',value)
    return value


def evaluate(folder,manifest):
    classical=manifest['method']=='classical'
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
    log=(folder/'driver.log').read_text() if (folder/'driver.log').exists() else ''
    audited=audit(folder) if log else {'calls':[],'observations':[],'violations':[]}
    session=session_evidence(folder) if log else {}
    if log:manifest['model_metadata']=session['metadata']
    latency=summarize(folder)
    original=benchmark_bag(folder,classical=classical) if (folder/'bag/metadata.yaml').exists() else {}
    if not log:save(folder/'driver_audit.json',audited)
    bag=verify(folder) if (folder/'bag/metadata.yaml').exists() else {}
    counts=bag.get('counts',{})
    common=['/clock','/scan','/map','/camera/image_raw','/odometry/filtered','/ground_truth/odometry','/cmd_vel','/tf']
    evidence=all(counts.get(t,0)>0 for t in common)
    if not classical:
        evidence=evidence and bool(audited['observations']) and not audited['violations'] and bag.get('all_observation_frames_in_bag') and not bag.get('movement_commands_missing_from_bag') and not bag.get('movement_terminal_statuses_missing_from_bag')
        if manifest['termination']=='driver_exit' and manifest.get('driver_exit_code') not in (None,0):
            manifest['termination']='infrastructure_error'
        actual=manifest.get('model_metadata',{})
        evidence=evidence and actual.get('actual_model')==manifest['model_requested'] and actual.get('effort')=='low'
    passed=original.get('passed',False) and manifest.get('completion_wall_s',math.inf)<=600
    outcome=classify(manifest['termination'],log,passed,evidence and manifest['cleanup_complete'])
    start=manifest.get('episode_started_monotonic_s',0);end=start+manifest.get('episode_wall_s',math.inf)
    selected=[e for e in events if start<=e['wall_s']<=end]
    poses=[e['data'] for e in selected if e['kind']=='truth']
    path=sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(poses,poses[1:]))
    coverage=[dict(elapsed_wall_s=e['wall_s']-start,**e['data']) for e in selected if e['kind']=='coverage']
    save(folder/'coverage.json',coverage)
    after_deadline=[e for e in events if e['wall_s']>start+manifest.get('wall_budget_s',600) and e['kind']=='velocity' and any(abs(v)>.001 for v in e['data'])]
    budget_tail=max((e['wall_s']-(start+manifest.get('wall_budget_s',600)) for e in after_deadline),default=0)
    interventions=0;previous=False
    for e in selected:
        if e['kind']=='obstacle':
            if e['data'] and not previous:interventions+=1
            previous=e['data']
    aborts=[e['data'] for e in selected if e['kind']=='status' and e['data']['state']=='aborted']
    usage=[]
    for line in log.splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if event.get('usage'):usage.append(event['usage'])
    result=dict(outcome=outcome,manifest=manifest,evidence_pass=bool(evidence),original_benchmark_pass=bool(passed),
        final_coverage=latency['final_coverage'],path_length_m=path,coverage_series='coverage.json',
        obstacle_intervention_definition='front guard blocked transitions (including invalid scan fail-closed stops)' if classical else 'executor aborts with obstacle reason',
        obstacle_interventions=interventions if classical else sum('obstacle' in a.get('reason','') for a in aborts),
        movement_aborts=aborts,movements=None if classical else sum(e['kind']=='command' and e['data']['action'] in ('drive','turn') for e in selected),
        nonzero_cmd_vel_after_wall_deadline_samples=len(after_deadline),last_nonzero_cmd_vel_after_wall_deadline_s=budget_tail,
        collision_count=None,collision_evidence='No contact sensor; aborts are not collisions',
        accessible_area_coverage=None,
        evaluator_caveat='Original evaluator labels whole-bag motion as after-completion when no completion transition exists; those post-completion diagnostics are not applicable to an incomplete episode.',model_request_latency_s=None,model_request_latency_reason='Not exposed by Codex CLI JSON; ready-to-request proxy in latency.json includes harness/network overhead',
        usage_events=usage,input_tokens=sum(u.get('input_tokens',0) for u in usage) if usage else None,
        output_tokens=sum(u.get('output_tokens',0) for u in usage) if usage else None,
        cached_input_tokens=sum(u.get('cached_input_tokens',0) for u in usage) if usage else None,
        reasoning_tokens=sum(u['reasoning_tokens'] for u in usage) if usage and all('reasoning_tokens' in u for u in usage) else None,
        latency=latency,bag=str(folder/'bag'),driver_log=str(folder/'driver.log') if log else None)
    if not classical and poses:
        from types import SimpleNamespace
        measured=infrastructure.evaluate(SimpleNamespace(poses=[e['data'] for e in events if e['kind']=='truth'], commands=[e['data'] for e in events if e['kind']=='command'], statuses=[e['data'] for e in events if e['kind']=='status'],velocity=bag.get('final_velocity')),audited)['actions']
        save(folder/'physical_actions.json',measured)
        result['maximum_turn_translation_m']=max((a.get('translation_m',0) for a in measured if a['command']['action']=='turn'),default=None)
    if session.get('total_token_usage'):
        total=session['total_token_usage']
        result.update(input_tokens=total.get('input_tokens'),output_tokens=total.get('output_tokens'),
            cached_input_tokens=total.get('cached_input_tokens'),reasoning_tokens=total.get('reasoning_output_tokens'),
            usage_source='session_evidence.json: cumulative token_count for this fresh thread; not summed cumulative snapshots',
            compaction_events=session['compaction_events'])
    if log:
        payloads=[]
        if (folder/'observations.jsonl').exists():
            for line in (folder/'observations.jsonl').read_text().splitlines():
                try:payloads.append(json.loads(line))
                except ValueError:continue
        if payloads:
            images=[c for c in payloads[-1]['content'] if c['type']=='image']
            for c,name in zip(images,('final-camera.jpg','final-online-map.png')):
                (folder/name).write_bytes(base64.b64decode(c['data']))
        observations=audited['observations']
        result['movement_action_counts']={kind:sum(e['kind']=='command' and e['data']['action']==kind for e in selected) for kind in ('drive','turn')}
        result['observation_count']=len(observations)
        result['navigation_history_lengths']=[len(o.get('navigation_history',[])) for o in observations]
        result['final_lidar_sectors']=(observations[-1].get('lidar',{}).get('sectors') if observations else None)
        result['finish_summaries']=[c.get('arguments',{}).get('summary') for c in audited['calls'] if c.get('tool')=='finish']
        result['post_action_timestamp_ordering']=dict(checked=len(audited['post_action_ordering']),all_ordered=all(o['ordered'] for o in audited['post_action_ordering']))
        result['sensor_states']={key:{state:sum(o.get(key,{}).get('state')==state for o in observations) for state in ('valid','stale','missing','invalid')} for key in ('lidar','slam_map','estimated_pose')}
    if (folder/'gui_evidence.json').exists():result['gui_evidence']=json.loads((folder/'gui_evidence.json').read_text())
    result['command_latency_unavailable_reason']='Native classical Twist/path messages lack correlation IDs; no heuristic matching claimed' if classical else None
    save(folder/'manifest.json',manifest);save(folder/'results.json',result)


def report(root):
    results={m:json.loads((root/m/'results.json').read_text()) for m in ('classical','llm') if (root/m/'results.json').exists()}
    save(root/'results.json',results)
    def fmt(v):
        return 'unavailable / N/A' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
    def value(r,key):
        if key in ('episode_wall_s','episode_sim_s','real_time_factor'):return r.get('manifest',{}).get(key)
        if key=='known_grid_percent':return (r.get('final_coverage') or {}).get('known_map_percent')
        if key=='frontier_components':return (r.get('final_coverage') or {}).get('unresolved_components')
        return r.get(key)
    rows=['# Paired exploration pilot', '',
        'One attempt per method cannot establish general superiority. An infrastructure failure leaves no valid performance comparison.', '',
        '| Measurement | Classical custom frontier/controller | gpt-5.6-luna / low |', '|---|---|---|']
    for key in ('outcome','known_grid_percent','frontier_components','episode_wall_s','episode_sim_s','real_time_factor','path_length_m','obstacle_interventions','movements','maximum_turn_translation_m','collision_count'):
        rows.append('| '+key+' | '+' | '.join(fmt(value(results.get(m,{}),key)) for m in ('classical','llm'))+' |')
    rows+=['', 'Known grid means known cells / rectangular SLAM grid, not accessible-area coverage. Completion and original hard-gate verdicts are recorded per episode. No contact sensor exists; zero obstacle aborts does not establish collision-free operation.', '',
        'Classical diagnostic: '+str(results.get('classical',{}).get('manifest',{}).get('error','none'))+'. No episode is retried.', '',
        'Driver finish summaries: '+json.dumps(results.get('llm',{}).get('finish_summaries',[]))+'. These are model claims, not evaluator findings. Raw action abort reasons and measured turn translation are retained in the results.', '',
        '## Latency and usage', '',
        '| Measurement (wall seconds) | Luna median | Luna p95 |', '|---|---|---|']
    llm=results.get('llm',{});latency=llm.get('latency',{});dist=latency.get('distributions',{})
    for label,key in [('Model-request latency (not exposed)',None),('Observation ready -> next request (harness/network/model proxy)','observation_ready_to_request_s'),('Command submission -> corresponding first nonzero cmd_vel','command_delivery_s'),('Action terminal -> next observation ready','feedback_delay_s')]:
        d=dist.get(key,{})
        rows.append('| '+label+' | '+fmt(d.get('median'))+' | '+fmt(d.get('p95'))+' |')
    rows+=['', 'Total ready-to-request waiting proxy: '+fmt(latency.get('observation_ready_to_request_total_s'))+' s. Classical native Twist/path messages lack action IDs, so exact delivery/feedback attribution is unavailable; no nearest-message estimate is claimed.', '',
        '| Exposed Luna tokens | Count |','|---|---|']
    for key in ('input_tokens','cached_input_tokens','output_tokens','reasoning_tokens'):
        rows.append('| '+key+' | '+fmt(llm.get(key))+' |')
    rows+=['', 'Cached input is included in input; reasoning is included in output. No subscription dollar cost is inferred. `session_evidence.json` preserves response IDs, UTC usage timestamps and cumulative totals; monotonic per-inference request/response pairs are unavailable.', '',
        '## Verification and artifacts', '',
        'Onboard observations: '+fmt(llm.get('observation_count'))+'. Timestamp checks: '+json.dumps(llm.get('post_action_timestamp_ordering'))+'. Sensor states: '+json.dumps(llm.get('sensor_states'))+'. Session compactions: '+str(len(llm.get('compaction_events',[])))+'. Full transcript and compact action ledger are retained.', '',
        'Per-method evidence pass: '+json.dumps({m:r.get('evidence_pass') for m,r in results.items()})+'. Owned cleanup complete: '+json.dumps({m:r.get('manifest',{}).get('cleanup_complete') for m,r in results.items()})+'. Bag verification contains actual final cmd_vel and every observed-frame/command/status match. GUI and independent host checks, when available, are saved separately.', '',
        'Final observed lidar sectors (meters): '+json.dumps(llm.get('final_lidar_sectors'))+'. These clearances do not prove frontier reachability or unreachability.', '',
        'Focused local checks cover sensor rendering/invalid data, stop interruption, frame ordering, movement-cap rejection, guard behavior and launch isolation. Build and run identifiers are saved with the episode sources.', '',
        '- [Machine-readable pair](results.json); available local documentation is saved with source snapshots.',
        '- [Classical results](classical/results.json), [original evaluator](classical/original_benchmark.json), bag `classical/bag/`, controller logs `classical/{frontier_detector,path_follower,obstacle_guard}.log`.',
        '- [Luna results](llm/results.json), [original evaluator](llm/original_benchmark.json), bag `llm/bag/`, driver `llm/driver.log`, exact observations `llm/observations.jsonl`, [final onboard map](llm/final-online-map.png).',
        '- Per method: `coverage.json`, `measurements.png`, `latency.json`, `action_timeline.json`, `bag_verification.json`, `cleanup.json`; Luna also `physical_actions.json`, `session_evidence.json`.',
        '- Classical execution sources: `source/`, `source_hashes.json`, `versions.json`. A resumed unstarted half uses `continuation/source/`, `continuation/source_hashes.json`, `continuation/versions.json`; otherwise it uses the same root snapshot. Postprocessing revisions: `reporting_source_hashes.json`.',
        '- Actual model/settings: `llm/model_metadata.json`; exact tool configuration and command: `llm/driver_config.json`, `llm/driver_command.json`. Base Git revision is in `versions.json`; working-tree hashes are authoritative.', '',
        'The original evaluator emits after-completion motion diagnostics even when no completion transition exists; these are not evidence of movement following a real completion. Its raw verdict is retained unchanged. Simulator IMU noise was not fixed by an explicit seed, so replaying the command does not promise identical trajectories.', '',
        ('**Next experiment:** repeat this same single pair after the startup fix, before changing worlds or models. A valid classical episode is missing from this comparison.' if any(r.get('outcome')=='infrastructure_error' for r in results.values()) else '**Next experiment:** repeat the fixed pair to measure run-to-run variation before expanding maps or models.')]
    (root/'report.md').write_text('\n'.join(rows)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method', choices=['both','classical','llm'], default='both')
    parser.add_argument('--resume-unstarted',action='store_true',help='Run only the never-started LLM half; requires classical results and no LLM directory. Never retries an episode.')
    parser.add_argument('--world',choices=['kd_world','office_loop_01'],default='kd_world')
    parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--headless',action='store_true');parser.add_argument('--model',default='gpt-5.6-luna')
    parser.add_argument('--hierarchical-selector',choices=['classical','mock','llm'])
    parser.add_argument('--wall-budget',type=int,choices=[120,600],default=600)
    parser.add_argument('--enable-model',action='store_true',help='Explicitly enable a future authorized hierarchical model episode')
    parser.add_argument('--stop-after-arrival',action='store_true',help='Mock diagnostic only: stop after first native goal arrival')
    args=parser.parse_args()
    if args.hierarchical_selector and args.method!='classical': parser.error('hierarchical runner uses --method classical for the shared motion/evaluator stack')
    if args.hierarchical_selector=='llm' and not args.enable_model: parser.error('model_execution_disabled: --enable-model required')
    if args.hierarchical_selector=='llm' and (args.model!='gpt-5.6-luna' or args.wall_budget!=600): parser.error('hierarchical model contract requires gpt-5.6-luna/low and 600 wall seconds')
    if args.stop_after_arrival and args.hierarchical_selector!='mock': parser.error('--stop-after-arrival is mock diagnostic only')
    methods=selected_methods(args.method,args.resume_unstarted)
    if os.environ.get('ROS_DOMAIN_ID')!='87':raise RuntimeError('Use scripts/run_exploration_pilot')
    args.output=args.output.resolve()
    if args.resume_unstarted:
        prior=json.loads((args.output/'classical/manifest.json').read_text())
        if (args.output/'llm').exists() or not prior.get('cleanup_complete'):
            raise RuntimeError('Continuation requires an unstarted LLM and verified classical cleanup')
    else:args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'COLCON_IGNORE').touch()
    lock=open('/tmp/rover-experiment-domain-87.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot=args.output/'continuation' if args.resume_unstarted else args.output
    snapshot.mkdir(exist_ok=True)
    configs={}
    for p in (Path.home()/'.codex/config.toml',ROOT/'.codex/config.toml'):
        if p.exists():configs[str(p)]=tomllib.loads(p.read_text()).get('mcp_servers',{})
    save(snapshot/'existing_mcp.json',configs)
    subprocess.run(['ps','-eo','pid,ppid,pgid,comm'],stdout=(snapshot/'existing_processes.txt').open('w'),check=True)
    source=list((ROOT/'src').rglob('*.py'))+list((ROOT/'src').rglob('*.yaml'))+list((ROOT/'src').rglob('package.xml'))
    source+=list((ROOT/'scripts').glob('*.py'))+[ROOT/'scripts/run_exploration_pilot',ROOT/'scripts/rover_driver_mcp',ROOT/'scripts/run_hierarchical_pilot',ROOT/'scripts/run_hierarchical_comparison',ROOT/'src/rover_description/worlds'/f'{args.world}.sdf',ROOT/'src/rover_description/urdf/rover.urdf.xacro']
    hashes={}
    # Optional local documentation is not required by a clean checkout.
    source+=list(ROOT.glob('HIERARCHICAL_INPUT_SPEC_v*.md'))+list(ROOT.glob('*CONTRACT*.md'))
    for p in source:
        relative=p.relative_to(ROOT);hashes[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
        dest=snapshot/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(snapshot/'source_hashes.json',hashes)
    if (ROOT/'BENCHMARK_CONTRACT.md').exists():
        shutil.copy2(ROOT/'BENCHMARK_CONTRACT.md',snapshot/'BENCHMARK_CONTRACT.md')
    save(snapshot/'versions.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        codex=subprocess.check_output(['codex','--version'],text=True).strip(),ros=os.environ.get('ROS_DISTRO'),
        model=args.model if 'llm' in methods else None,effort='low' if 'llm' in methods else None,methods=methods,python=os.sys.version))
    versions=json.loads((snapshot/'versions.json').read_text())
    packages=subprocess.run(['dpkg-query','-W','ros-jazzy-slam-toolbox','ros-jazzy-robot-localization','ros-jazzy-ros-gz-sim'],capture_output=True,text=True)
    versions['system_packages']=packages.stdout;versions['package_version_error']=packages.stderr or None
    save(snapshot/'versions.json',versions)
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def interrupt(sig,frame):
        global INTERRUPTED
        if INTERRUPTED or CLEANING_UP:return  # Never interrupt owned teardown.
        INTERRUPTED=True
        if ACTIVE_CANCEL_PATH is not None:latch_cancel(ACTIVE_CANCEL_PATH)
        raise InterruptedError('signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    try:
        for method in methods:
            manifest=run_episode(args.output/method,method,args);report(args.output)
            if manifest['termination'] in ('interrupted','infrastructure_error') or not manifest['cleanup_complete']:break
    finally:
        report(args.output)
        if args.hierarchical_selector:
            from hierarchical_results import report as hierarchy_report
            hierarchy_report(args.output)
        rclpy.try_shutdown()
    print('Pilot artifacts:',args.output,flush=True)


if __name__=='__main__':main()
