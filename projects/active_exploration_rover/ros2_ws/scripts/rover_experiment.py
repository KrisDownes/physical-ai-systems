"""Reproducible camera-only model episodes; ground truth stays in this supervisor."""
import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
import yaml
from types import SimpleNamespace
from PIL import Image as PILImage, ImageChops, ImageStat

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from verify_rover_bags import verify as verify_bag

ROOT = Path(__file__).resolve().parents[1]
VERIFY_TOOLS = False
OPTIONS = None
from exploration_measurement import attach, summarize, announce_completion, benchmark_bag
TOPICS = ['/clock', '/camera/image_raw', '/camera/camera_info', '/scan',
          '/odometry/filtered', '/ground_truth/odometry', '/cmd_vel',
          '/agent_command', '/agent_status', '/imu/data_raw', '/agent_timing', '/map', '/tf', '/tf_static', '/exploration_complete', '/exploration_result']
TASK = '''Navigate using only the rover MCP tools observe, drive, turn, stop, finish.
Your goal is to approach the red box, face it, and stop with approximately
0.65 to 0.95 meters between the rover center and the nearest box surface.
Use the camera to estimate distance; exact measurement is not available.
Observe first. Choose every navigation action yourself from the images.
After each movement, observe again and check the capture timestamp is later
than terminal_sim_time_s. Explain expected and observed image changes briefly.
Drive at most 0.5 m per action and turn at most 30 degrees per action; use smaller
actions near the box. Do not collide. At most 16 movement actions. If a movement
fails or is aborted, finish without retrying it. Call finish when done or uncertain.
Do not use other tools, shell, files, maps, coordinates, or outside information.
Do not infer success solely from action status. Your final response should state
whether you believe you reached the goal and cite the visual evidence.'''
CALIBRATION = '''\nFixed calibration: the camera image is 320 by 240 pixels with a 90-degree
horizontal field of view (focal length approximately 160 pixels). The camera
is 0.225 meters forward of the rover center. The red box is 0.5 meters wide,
0.5 meters deep, and 1 meter tall. These are fixed dimensions, not a pose or map.
Use visible horizontal edges as well as vertical extent when estimating range;
the top can leave the image well before the rover reaches the target distance.
Ground-truth distance and success feedback are unavailable. Base every choice
and every claim of image change on the actual image.'''


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9


class Monitor(Node):
    def __init__(self, folder, publish_completion=True):
        super().__init__('experiment_supervisor')
        self.folder = folder
        self.stream = (folder / 'events.jsonl').open('w')
        self.latest = {}
        self.poses = []
        self.statuses = []
        self.commands = []
        self.velocity = None
        self.lock = threading.Lock()
        self.create_subscription(Odometry, '/ground_truth/odometry', self.pose, 10)
        self.create_subscription(Odometry, '/odometry/filtered', lambda m: self.seen('odom', stamp(m)), 10)
        self.create_subscription(Image, '/camera/image_raw', lambda m: self.seen('camera', stamp(m)), qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', lambda m: self.seen('scan', stamp(m)), qos_profile_sensor_data)
        self.create_subscription(String, '/agent_status', self.status, 10)
        self.create_subscription(String, '/agent_command', self.command, 10)
        self.create_subscription(Twist, '/cmd_vel', self.vel, 10)
        attach(self, publish_completion=publish_completion)
        self.stop_pub = self.create_publisher(String, '/agent_command', 10)

    def seen(self, key, value):
        self.latest[key] = (value, time.monotonic())

    def event(self, kind, data):
        with self.lock:
            self.stream.write(json.dumps({'kind': kind, 'wall_s': time.monotonic(), 'data': data}) + '\n')
            self.stream.flush()

    def pose(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        value = {'t': stamp(msg), 'x': p.x, 'y': p.y,
                 'yaw': math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))}
        self.poses.append(value)
        self.seen('truth', value['t'])
        self.event('truth', value)

    def status(self, msg):
        value = json.loads(msg.data)
        self.statuses.append(value)
        self.event('status', value)

    def command(self, msg):
        value = json.loads(msg.data)
        self.commands.append(value)
        self.event('command', value)

    def vel(self, msg):
        self.velocity = (msg.linear.x, msg.angular.z)
        self.seen('velocity', self.velocity)
        self.event('velocity', self.velocity)

    def ready(self):
        return all(k in self.latest and time.monotonic()-self.latest[k][1] < 2
                   for k in ('camera', 'scan', 'odom', 'truth', 'velocity'))

    def graph(self):
        return {'nodes': self.get_node_names(), 'velocity_publishers':
                [p.node_name for p in self.get_publishers_info_by_topic('/cmd_vel')],
                'command_publishers': [p.node_name for p in self.get_publishers_info_by_topic('/agent_command')],
                'raw_velocity_publishers': [p.node_name for p in self.get_publishers_info_by_topic('/cmd_vel_raw')],
                'guarded_velocity_publishers': [p.node_name for p in self.get_publishers_info_by_topic('/cmd_vel_guarded')]}


class Processes:
    def __init__(self, folder):
        self.folder, self.owned, self.cleanup = folder, [], []
        self.stopped_pids = set()

    def start(self, name, command, **kwargs):
        log = (self.folder / (name + '.log')).open('w')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, **kwargs)
        self.owned.append((name, proc, log))
        save(self.folder / 'processes.json', [{'name': n, 'pid': p.pid, 'pgid': p.pid}
                                             for n, p, _ in self.owned])
        return proc

    def stop(self, entry):
        name, proc, log = entry
        if proc.pid in self.stopped_pids:return
        sent = []
        for sig, duration in ((signal.SIGINT, 8), (signal.SIGTERM, 4), (signal.SIGKILL, 2)):
            try:
                os.killpg(proc.pid, sig)
                sent.append(sig.name)
            except ProcessLookupError:
                break
            end = time.monotonic() + duration
            while time.monotonic() < end:
                proc.poll()
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(.1)
            else:
                continue
            break
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
            gone = False
        except ProcessLookupError:
            gone = True
        log.close()
        if gone:self.stopped_pids.add(proc.pid)
        self.cleanup.append({'name': name, 'returncode': proc.returncode,
                             'signals': sent, 'group_gone': gone})

    def close(self):
        for entry in reversed(self.owned):
            self.stop(entry)
        save(self.folder / 'cleanup.json', self.cleanup)


def wait_for(predicate, timeout, label, processes=()):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('process exited while waiting for ' + label)
        if predicate():
            return
        time.sleep(.1)
    raise RuntimeError('timeout waiting for ' + label)


def driver_command(model, cwd):
    if VERIFY_TOOLS:
        return [str(ROOT/'.driver_venv/bin/python'), str(ROOT/'scripts/verify_rover_tools.py')], {'mode': 'fixed tool diagnostic, not model navigation'}
    config = {
        'model_reasoning_effort': 'low', 'web_search': 'disabled',
        'approval_policy': 'never', 'sandbox_mode': 'read-only',
        'features.shell_tool': False, 'features.view_image': False,
        'features.apps': False, 'features.plugins': False,
        'features.multi_agent': False, 'features.memories': False,
        'features.browser_use': False, 'features.computer_use': False,
        'features.image_generation': False, 'features.skill_search': False,
        'features.skip_host_skill_discovery': True,
        'mcp_servers.rover.command': str(ROOT / 'scripts/rover_driver_mcp'),
        'mcp_servers.rover.env.ROS_DOMAIN_ID': os.environ['ROS_DOMAIN_ID'],
        'mcp_servers.rover.env.ROVER_MIN_COMMAND_SUBSCRIBERS': '3',
        'mcp_servers.rover.enabled_tools': ['observe', 'drive', 'turn', 'stop', 'finish'],
        'mcp_servers.rover.default_tools_approval_mode': 'approve',
        'mcp_servers.rover.required': True,
        'mcp_servers.rover.startup_timeout_sec': 60,
        'mcp_servers.rover.tool_timeout_sec': 35,
    }
    cmd = ['codex', 'exec', '--ignore-user-config', '--json',
           '--skip-git-repo-check', '--model', model, '-C', str(cwd)]
    for key, value in config.items():
        cmd += ['-c', key + '=' + json.dumps(value)]
    return cmd + [TASK], config


def audit(folder):
    calls, violations, observations, images, messages, diagnostics = [], [], [], [], [], []
    for line in (folder / 'driver.log').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get('item', {})
        if event.get('type') in {'error', 'turn.failed'}:
            diagnostics.append(event)
        if event.get('type') != 'item.completed':
            continue
        kind = item.get('type')
        if kind == 'mcp_tool_call':
            calls.append(item)
            if item.get('server') != 'rover' or item.get('tool') not in {'observe','drive','turn','stop','finish'}:
                violations.append(item)
            result = item.get('result') or {}
            if item.get('error') or result.get('isError'):
                diagnostics.append({'tool': item.get('tool'), 'error': item.get('error'),
                                    'content': result.get('content', [])})
            for content in result.get('content', []):
                if content.get('type') == 'image':
                    path = folder / f'observation-{len(images):03}.jpg'
                    path.write_bytes(base64.b64decode(content['data']))
                    images.append(path.name)
                if content.get('type') == 'text':
                    try:
                        data = json.loads(content['text'])
                    except ValueError:
                        continue
                    if 'captured_sim_time_s' in data:
                        observations.append(data)
        elif kind == 'agent_message':
            messages.append(item.get('text', ''))
        elif kind == 'error':
            diagnostics.append(item)
        elif kind not in {'reasoning', 'plan', None}:
            violations.append(item)
    ordering = []
    for obs in observations:
        action = obs.get('last_completed_action_result')
        if action:
            ordering.append({'action_id': action['command_id'],
                             'capture': obs['captured_sim_time_s'],
                             'terminal': action['terminal_sim_time_s'],
                             'ordered': obs['captured_sim_time_s'] > action['terminal_sim_time_s']})
    image_evidence = []
    previous = None
    for name in images:
        current = PILImage.open(folder/name).convert('RGB')
        red = [(x, y) for y in range(current.height) for x in range(current.width)
               if (lambda rgb: rgb[0] > 50 and rgb[0] > 2*rgb[1] and rgb[0] > 2*rgb[2])(current.getpixel((x,y)))]
        bbox = ([min(p[0] for p in red), min(p[1] for p in red),
                 max(p[0] for p in red), max(p[1] for p in red)] if red else None)
        image_evidence.append({'image': name, 'red_bbox_xyxy': bbox,
            'mean_absolute_change': sum(ImageStat.Stat(ImageChops.difference(current, previous)).mean)/3
                                    if previous and previous.size == current.size else None})
        previous = current
    result = {'calls': calls, 'violations': violations, 'observations': observations,
              'images': images, 'messages': messages, 'diagnostics': diagnostics,
              'post_action_ordering': ordering,
              'image_evidence': image_evidence,
              'limitation': 'Visible completed-tool transcript audit; not proof of OS-level isolation.'}
    save(folder / 'driver_audit.json', result)
    return result


def model_metadata(folder):
    """Read only this fresh driver's session metadata, never another session."""
    thread_id = None
    for line in (folder / 'driver.log').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'thread.started':
            thread_id = event.get('thread_id')
    metadata = {'thread_id': thread_id, 'actual_model': None,
                'limitation': 'Upstream server snapshot is not exposed by the CLI.'}
    if thread_id:
        for path in (Path.home()/'.codex/sessions').rglob('*' + thread_id + '*.jsonl'):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event.get('type') == 'turn_context':
                    payload = event['payload']
                    metadata.update(actual_model=payload.get('model'),
                                    effort=payload.get('effort'),
                                    cwd=payload.get('cwd'),
                                    approval_policy=payload.get('approval_policy'))
    save(folder/'model_metadata.json', metadata)
    return metadata


def evaluate(monitor, audit_result):
    poses = list(monitor.poses)
    final = poses[-1]
    gap = math.hypot(max(abs(final['x']-2)-.25, 0), max(abs(final['y'])-.25, 0))
    bearing = math.atan2(-final['y'], 2-final['x'])
    facing = abs(math.atan2(math.sin(bearing-final['yaw']), math.cos(bearing-final['yaw'])))
    actions = []
    commands = list(monitor.commands)
    command_gaps = []
    for call in audit_result['calls']:
        if call.get('tool') not in {'drive', 'turn'}:
            continue
        for content in (call.get('result') or {}).get('content', []):
            if content.get('type') != 'text':
                continue
            try:
                result = json.loads(content['text'])
            except ValueError:
                continue
            identifier = result.get('command_id')
            if identifier and not any(c['id'] == identifier for c in commands):
                command_gaps.append(identifier)
                commands.append({'id': identifier, 'action': call['tool'], **call['arguments']})
    for command in commands:
        if command['action'] == 'stop':
            continue
        statuses = [s for s in monitor.statuses if s['id'] == command['id']]
        accepted = next((s for s in statuses if s['state'] == 'accepted'), None)
        terminal = next((s for s in statuses if s['state'] in {'succeeded','aborted','rejected'}), None)
        value = {'command': command, 'terminal': terminal}
        if accepted and terminal:
            before = min(poses, key=lambda p: abs(p['t']-accepted['sim_time_s']))
            next_start = min((s['sim_time_s'] for s in monitor.statuses
                              if s['state'] == 'accepted' and s['sim_time_s'] > terminal['sim_time_s']),
                             default=math.inf)
            end = min(terminal['sim_time_s']+.2, next_start)
            candidates = [p for p in poses if p['t'] <= end]
            after = candidates[-1]
            value.update(translation_m=math.hypot(after['x']-before['x'], after['y']-before['y']),
                         rotation_deg=math.degrees(math.atan2(math.sin(after['yaw']-before['yaw']), math.cos(after['yaw']-before['yaw']))),
                         measured_from_sim_s=before['t'], measured_to_sim_s=after['t'])
        actions.append(value)
    finished = False
    for call in audit_result['calls']:
        if call.get('tool') == 'finish' and not call.get('error'):
            for content in (call.get('result') or {}).get('content', []):
                if content.get('type') == 'text':
                    try:
                        finish_result = json.loads(content['text'])
                        finished = (finish_result.get('finished') is True
                                    and finish_result.get('stop_result', {}).get('terminal_state') == 'succeeded')
                    except ValueError:
                        pass
    ordered_ids = {o['action_id'] for o in audit_result['post_action_ordering'] if o['ordered']}
    actions_verified = bool(actions) and all(
        a['terminal'] and a['terminal']['state'] == 'succeeded'
        and a['command']['id'] in ordered_ids for a in actions)
    result = {'final_pose': final, 'box_surface_distance_m': gap,
              'facing_error_deg': math.degrees(facing), 'actions': actions,
              'final_velocity': monitor.velocity, 'driver_called_finish': finished,
              'all_movements_have_success_and_later_frame': actions_verified,
              'command_recording_gaps': command_gaps,
              'success': .65 <= gap <= .95 and facing <= math.radians(15) and finished
                         and actions_verified and not command_gaps
                         and monitor.velocity == (0., 0.) and not audit_result['violations'],
              'collision_measurement': 'No contact sensor; collision absence is not established.'}
    return result


def episode(folder, model, yaw, timeout):
    folder.mkdir(parents=True)
    processes = Processes(folder)
    monitor = None
    spin = None
    scenarios = {'nominal': (0., 0., 0.), 'translated_south': (0., -2., 0.),
                 'translated_north': (0., 2., 0.), 'south_yaw90': (0., -2., math.pi/2)}
    spawn_x, spawn_y, spawn_yaw = scenarios[OPTIONS.scenario]
    if OPTIONS.task == 'red-box':
        spawn_yaw += yaw
    world_path = ROOT/'src/rover_description/worlds'/f'{OPTIONS.world}.sdf'
    manifest = {'model_requested': model, 'reasoning_effort': 'low', 'ros_domain_id': 87,
                'runner_pid': os.getpid(),
                'mode': 'tool_diagnostic' if VERIFY_TOOLS else 'model_navigation',
                'task_type': OPTIONS.task, 'wall_budget_s': timeout, 'action_budget': OPTIONS.action_budget,
                'original_wall_deadline_s': 600,
                'world': OPTIONS.world, 'world_sha256': hashlib.sha256(world_path.read_bytes()).hexdigest(),
                'scenario': OPTIONS.scenario,
                'spawn': {'x': spawn_x, 'y': spawn_y, 'yaw': spawn_yaw}, 'task': TASK, 'started_unix_s': time.time()}
    try:
        probe = Node('experiment_preflight')
        for _ in range(20):
            rclpy.spin_once(probe, timeout_sec=.1)
        existing = [name for name in probe.get_node_names() if name != 'experiment_preflight']
        probe.destroy_node()
        manifest['preexisting_domain_nodes'] = existing
        if existing:
            raise RuntimeError('experiment ROS domain already occupied: ' + ', '.join(existing))
        env = dict(os.environ, GZ_PARTITION='rover-experiment-' + folder.parent.name + '-' + folder.name,
                   ROVER_MIN_COMMAND_SUBSCRIBERS='3')
        simulation = processes.start('simulation', ['ros2', 'launch', 'rover_exploration',
            'exploration.launch.py', 'agent_mode:=true', f'enable_rviz:={str(OPTIONS.gui).lower()}',
            'gazebo_args:=-r -s --headless-rendering', f'spawn_yaw:={spawn_yaw}',
            f'spawn_x:={spawn_x}', f'spawn_y:={spawn_y}', f'world_file:={world_path}'], env=env)
        manifest['gazebo_partition'] = env['GZ_PARTITION']
        if OPTIONS.gui:
            processes.start('gazebo_gui', ['gz', 'sim', '-g', '-v', '4'], env=env)
            processes.start('camera_gui', ['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '/camera/image_raw'], env=env)
        if OPTIONS.live_stats and OPTIONS.gui:
            processes.start('live_view', ['/usr/bin/python3', str(ROOT/'scripts/rover_live_view.py'), str(folder)], env=env)
        monitor = Monitor(folder)
        executor = SingleThreadedExecutor()
        executor.add_node(monitor)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()
        wait_for(monitor.ready, 60, 'fresh sensors', [simulation])
        manifest['graph'] = monitor.graph()
        if manifest['graph']['velocity_publishers'] != ['agent_executor']:
            raise RuntimeError('ambiguous velocity ownership')
        if manifest['graph']['command_publishers'] != ['experiment_supervisor']:
            raise RuntimeError('existing driver connection on experiment domain')
        recorder = processes.start('recorder', ['ros2', 'bag', 'record', '-o', str(folder/'bag'), *TOPICS], env=env)
        wait_for(lambda: 'Listening for topics' in (folder/'recorder.log').read_text(), 20, 'recorder', [recorder])
        with tempfile.TemporaryDirectory(prefix='rover-driver-') as empty:
            cmd, config = driver_command(model, Path(empty))
            manifest.update(driver_command=cmd, driver_settings=config, driver_cwd=empty)
            save(folder / 'manifest.json', manifest)
            driver = processes.start('driver', cmd, cwd=empty, env=env, stdin=subprocess.DEVNULL)
            started = time.monotonic()
            last_print = 0
            while driver.poll() is None:
                elapsed = time.monotonic()-started
                if simulation.poll() is not None or recorder.poll() is not None:
                    raise RuntimeError('simulation or recording process exited')
                if OPTIONS.task == 'explore-map' and monitor.completed:
                    manifest['termination'] = 'coverage_completion'
                    manifest['completion_wall_s'] = elapsed
                    # Remove the command producer before declaring completion.
                    processes.stop(processes.owned.pop())
                    monitor.stop_pub.publish(String(data=json.dumps({'id':'supervisor-complete','action':'stop'})))
                    wait_for(lambda: monitor.velocity == (0.,0.), 2, 'completion stop')
                    time.sleep(.5)
                    announce_completion(monitor)
                    break
                movements = sum(c['action'] in ('drive','turn') for c in monitor.commands)
                if elapsed >= timeout or movements >= OPTIONS.action_budget:
                    manifest['termination'] = 'wall_budget' if elapsed >= timeout else 'action_budget'
                    break
                if OPTIONS.live_stats and elapsed-last_print >= 1:
                    c = monitor.commands[-1] if monitor.commands else {}
                    obs = next((e for e in reversed(monitor.timing_events) if e['kind']=='observation_ready'),None)
                    state = 'executing' if monitor.velocity and any(monitor.velocity) else 'waiting/stopped'
                    sim = monitor.latest.get('truth',(0,))[0]
                    line = f"wall={elapsed:.1f}s sim={sim:.1f}s RTF={sim/max(elapsed,.001):.2f} {state} action={c} observation_age={time.monotonic()-obs['received_wall_s'] if obs else None} coverage={monitor.coverage} target=0 components >=5 cells for 8 sim seconds"
                    print(line, flush=True)
                    (folder/'live-status.txt').write_text(line)
                    last_print=elapsed
                time.sleep(.05)
            if driver.poll() is None:
                processes.stop(processes.owned.pop())
            manifest['mission_wall_s'] = time.monotonic()-started
            if OPTIONS.gui:
                windows = subprocess.run(['xwininfo','-root','-tree'],capture_output=True,text=True)
                (folder/'desktop-windows.txt').write_text(windows.stdout+windows.stderr)
                manifest['visible_window_evidence'] = 'Gazebo' in windows.stdout
            if OPTIONS.task == 'explore-map' and monitor.completed:
                end_sim = monitor.latest['truth'][0]+5
                wait_for(lambda: monitor.latest['truth'][0]>=end_sim, 30, 'post completion stopping observation', [simulation])
            manifest['driver_exit_code'] = driver.returncode
        time.sleep(.5)
        result = audit(folder)
        manifest['model_metadata'] = model_metadata(folder)
        save(folder / 'evaluation.json', evaluate(monitor, result))
        if driver.returncode != 0 and not manifest.get('termination'):
            raise RuntimeError(f'driver exited {driver.returncode}: ' + json.dumps(result['diagnostics']))
    except (Exception, KeyboardInterrupt) as error:
        manifest['error'] = str(error)
        if isinstance(error, (InterruptedError, KeyboardInterrupt)):
            manifest['termination'] = 'interrupted'
    finally:
        if processes.owned and processes.owned[-1][0] == 'driver':
            processes.stop(processes.owned.pop())
        if monitor:
            monitor.stop_pub.publish(String(data=json.dumps({'id': 'supervisor-cleanup', 'action':'stop'})))
            time.sleep(.3)
        processes.close()
        if manifest.get('driver_exit_code') is None:
            manifest['driver_exit_code'] = next((p['returncode'] for p in processes.cleanup
                                                 if p['name'] == 'driver'), None)
        bag_metadata = folder/'bag/metadata.yaml'
        if bag_metadata.exists():
            metadata = yaml.safe_load(bag_metadata.read_text())['rosbag2_bagfile_information']
            counts = {topic['topic_metadata']['name']:topic['message_count']
                      for topic in metadata['topics_with_message_count']}
            manifest['bag_topic_counts'] = counts
            manifest['bag_complete'] = all(counts.get(topic, 0) > 0 for topic in TOPICS if topic != '/exploration_result')
        else:
            manifest['bag_complete'] = False
        if (folder/'driver.log').exists() and not (folder/'evaluation.json').exists():
            try:
                partial_audit = audit(folder)
                if monitor and monitor.poses:
                    save(folder/'evaluation.json', evaluate(monitor, partial_audit))
            except Exception as error:
                manifest['partial_evaluation_error'] = str(error)
        if bag_metadata.exists() and (folder/'driver_audit.json').exists():
            try:
                evidence = verify_bag(folder)
                manifest['bag_evidence_complete'] = (
                    evidence['all_observation_frames_in_bag']
                    and not evidence['movement_commands_missing_from_bag']
                    and not evidence['movement_terminal_statuses_missing_from_bag'])
            except Exception as error:
                manifest['bag_evidence_complete'] = False
                manifest['bag_verification_error'] = str(error)
        if monitor:
            executor.shutdown()
            spin.join(timeout=2)
            monitor.destroy_node()
            monitor.stream.close()
        if monitor:
            try:
                metrics = summarize(folder)
                if OPTIONS.task == 'explore-map' and not VERIFY_TOOLS:
                    original = benchmark_bag(folder) if bag_metadata.exists() else {}
                    save(folder/'evaluation.json', exploration_verdict(manifest, metrics, original,
                        json.loads((folder/'driver_audit.json').read_text()) if (folder/'driver_audit.json').exists() else {}))
            except Exception as error:
                manifest['evaluation_error'] = str(error)
                save(folder/'evaluation.json', {'success': False, 'termination': 'evaluation_error', 'error': str(error)})
        manifest['ended_unix_s'] = time.time()
        save(folder / 'manifest.json', manifest)
        write_report(folder)
    return manifest


def exploration_verdict(manifest, metrics, original, audited):
    coverage = metrics.get('final_coverage') or {}
    complete = bool(coverage.get('completed'))
    termination = manifest.get('termination') or ('driver_service_error' if manifest.get('driver_exit_code', 0) != 0
                                                 else 'driver_early_exit')
    timing_pass = complete and manifest.get('completion_wall_s', float('inf')) <= 600
    benchmark_pass = bool(original.get('passed')) and timing_pass
    evidence_pass = (bool(audited.get('calls')) and not audited.get('violations')
                     and manifest.get('bag_complete', False) and manifest.get('bag_evidence_complete', False))
    return dict(eventual_coverage_completion=complete, final_coverage=coverage,
                original_benchmark_pass=benchmark_pass, original_timing_pass=timing_pass,
                evidence_pass=evidence_pass, success=benchmark_pass and evidence_pass and not manifest.get('error'),
                termination=termination, latency=metrics,
                limitation='Coverage is known rectangular grid cells, not reachable-area coverage. Original hard gates are in original_benchmark.json; no contact sensor.')


def write_report(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    evaluation = json.loads((folder/'evaluation.json').read_text()) if (folder/'evaluation.json').exists() else {}
    audit_result = json.loads((folder/'driver_audit.json').read_text()) if (folder/'driver_audit.json').exists() else {}
    cleanup = json.loads((folder/'cleanup.json').read_text())
    success = evaluation.get('success', False) and not manifest.get('error')
    if manifest.get('task_type') == 'explore-map':
        (folder/'report.md').write_text('# Full-map exploration attempt\n\n' + json.dumps(evaluation, indent=2) + '\n\nManifest: ' + json.dumps(manifest, indent=2) + '\n\nCleanup: ' + json.dumps(cleanup, indent=2))
        return
    diagnostic = manifest.get('mode') == 'tool_diagnostic'
    if diagnostic:
        success = manifest.get('driver_exit_code') == 0 and not manifest.get('error')
    success = success and manifest.get('bag_complete', False) and manifest.get('bag_evidence_complete', False)
    success = success and all(p['group_gone'] for p in cleanup)
    outcome = 'PASS' if success else ('INCOMPLETE' if manifest.get('error') else 'FAIL')
    lines = [f'# Rover {"tool diagnostic" if diagnostic else "episode"}: {outcome}', '',
             'No model was called; these are fixed transport diagnostics.' if diagnostic else
             f'Model requested: `{manifest["model_requested"]}`; actual harness model: '
             f'`{manifest.get("model_metadata", {}).get("actual_model", "not recorded")}`; effort: low.',
             f'ROS domain: {manifest["ros_domain_id"]}. Spawn: {manifest["spawn"]}.', '',
             'Diagnostic criterion: interrupt a pending drive, complete a small turn, verify later frames, and finish.' if diagnostic else
             'Criterion: final rover-center distance to the red box surface 0.65–0.95 m, '
             'facing error ≤15°, driver finish, zero command velocity, and no observed tool violations.', '',
             f'Final box surface distance: {evaluation.get("box_surface_distance_m")} m.',
             f'Facing error: {evaluation.get("facing_error_deg")} degrees.',
             f'Driver called finish: {evaluation.get("driver_called_finish")}. '
             f'Final velocity: {evaluation.get("final_velocity")}.',
             f'Tool audit violations: {len(audit_result.get("violations", []))}. '
             f'Post-action frame ordering: {audit_result.get("post_action_ordering", [])}.', '',
             '| Action | Requested | Terminal | Translation (m) | Rotation (deg) |',
             '|---|---|---|---|---|']
    for a in evaluation.get('actions', []):
        c = a['command']
        lines.append(f'| {c["id"]} | {c} | {a.get("terminal")} | {a.get("translation_m")} | {a.get("rotation_deg")} |')
    lines += ['', 'Image measurements: see `driver_audit.json` and saved observation JPEGs.',
              'Full recording: `bag/`; model-visible events: `driver.log`; independent ground truth: `events.jsonl`.',
              'Isolation is supported by tool configuration and a visible transcript audit, not an OS-level access proof.',
              'No contact sensor is present; collision absence is not established.',
              f'All owned process groups gone: {all(p["group_gone"] for p in cleanup)}.',
              f'Bag contains all required topics: {manifest.get("bag_complete", "not checked")}.',
              f'Bag contains every observed frame and movement record: {manifest.get("bag_evidence_complete", "not checked")}.',
              f'Commands missing from the ROS event recording: {evaluation.get("command_recording_gaps", [])}.',
              f'Execution error: {manifest.get("error", "none")}.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    global TASK, VERIFY_TOOLS, OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'experiment_runs'/time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--world', choices=['kd_world', 'open_arena'], default='kd_world')
    parser.add_argument('--scenario', choices=['nominal', 'translated_south', 'translated_north', 'south_yaw90'], default='nominal')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--action-budget', type=int, default=300)
    parser.add_argument('--task', choices=['explore-map','red-box'], default='explore-map')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--live-stats', action='store_true')
    parser.add_argument('--uncalibrated', action='store_true', help='Reproduce the original task without fixed calibration')
    parser.add_argument('--verify-tools', action='store_true', help='Run fixed MCP interruption and turn diagnostics, not model navigation')
    parser.add_argument('--reevaluate', type=Path, help='Reevaluate saved events and transcripts without simulation or model calls')
    args = parser.parse_args()
    OPTIONS = args
    if args.timeout <= 0 or args.action_budget <= 0: parser.error("budgets must be positive")
    VERIFY_TOOLS = args.verify_tools
    if VERIFY_TOOLS:
        args.episodes = 1
    if args.reevaluate:
        for folder in sorted(args.reevaluate.glob('episode-*')):
            reevaluate(folder)
        return
    if args.task == 'explore-map':
        TASK = f"""Explore the entire accessible environment using only rover observe, drive, turn, stop, finish tools.
Observe first and after every movement. Choose all exploration decisions from camera images and your own visual memory.
Seek unseen areas and remember routes and landmarks. You receive no evaluator map, pose, coverage, or ground truth.
An independent supervisor determines completion; do not declare success yourself or stop after a demonstration.
Continue exploring until supervisor stops you, an unrecoverable safety problem occurs, or {args.action_budget} movements / {args.timeout} wall seconds are exhausted.
Use the existing bounds: drive at most 0.5m, turn at most 30 degrees per action. Never weaken obstacle protection.
After an obstacle abort, observe and choose a different safe action; finish only if no safe continuation exists.
Check each post-movement frame is newer than terminal_sim_time_s. Keep responses concise and action-focused.
Do not use other tools, shell, files, maps, world coordinates, or outside information."""
    elif not args.uncalibrated:
        TASK += CALIBRATION
    if os.environ.get('ROS_DOMAIN_ID') != '87':
        raise RuntimeError('Use scripts/run_rover_experiment (ROS_DOMAIN_ID=87)')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = open('/tmp/rover-experiment-domain-87.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    configurations = {}
    for path in (Path.home()/'.codex/config.toml', ROOT/'.codex/config.toml'):
        if path.exists():
            configurations[str(path)] = tomllib.loads(path.read_text()).get('mcp_servers', {})
    save(args.output/'existing_mcp.json', configurations)
    subprocess.run(['ps', '-eo', 'pid,ppid,pgid,comm'], stdout=(args.output/'existing_processes.txt').open('w'), check=True)
    source_files = list((ROOT/'src/visual_rover_agent/visual_rover_agent').glob('*.py')) + [Path(__file__),
        ROOT/'scripts/run_rover_experiment', ROOT/'scripts/rover_driver_mcp',
        ROOT/'scripts/verify_rover_tools.py', ROOT/'scripts/verify_rover_bags.py', ROOT/'scripts/exploration_measurement.py',
        ROOT/'src/rover_description/urdf/rover.urdf.xacro', ROOT/'src/rover_description/worlds/kd_world.sdf',
        ROOT/'src/rover_description/worlds'/f'{args.world}.sdf',
        ROOT/'src/rover_description/launch/sim.launch.py', ROOT/'src/rover_exploration/launch/exploration.launch.py',
        ROOT/'src/rover_exploration/config/ekf.yaml']
    save(args.output/'source_hashes.json', {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    for path in source_files:
        destination = args.output/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    versions = {'codex': subprocess.check_output(['codex','--version'], text=True).strip(),
                'ros_distro': os.environ.get('ROS_DISTRO'), 'python': os.sys.version}
    save(args.output/'versions.json', versions)
    rclpy.init()
    def interrupted(signum, _frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        results = []
        for i in range(args.episodes):
            folder = args.output/f'episode-{i+1:02}'
            print('Starting', folder, flush=True)
            manifest = episode(folder, args.model, (0., .35, -.35)[i % 3], args.timeout)
            results.append(manifest)
            print('Completed', folder, manifest.get('error', 'recorded'), flush=True)
            if manifest.get('error'):
                break
            cleanup = json.loads((folder/'cleanup.json').read_text())
            if not all(p['group_gone'] for p in cleanup):
                break
        save(args.output/'results.json', results)
        (args.output/'report.md').write_text('# Rover experiment\n\n' + '\n'.join(
            f'- [Episode {i+1}](episode-{i+1:02}/report.md)' for i in range(len(results))) + '\n')
    finally:
        rclpy.shutdown()
    print('Artifacts:', args.output, flush=True)
    if any(r.get('error') or not r.get('bag_complete') or not r.get('bag_evidence_complete') for r in results):
        raise SystemExit(1)


def reevaluate(folder):
    for name in ('evaluation.json', 'report.md', 'latency.json', 'original_benchmark.json'):
        path = folder/name
        backup = folder/(name + '.before-restart')
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
    events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    values = lambda kind: [e['data'] for e in events if e['kind'] == kind]
    monitor = SimpleNamespace(poses=values('truth'), commands=values('command'),
                              statuses=values('status'), velocity=tuple(values('velocity')[-1]))
    audited = audit(folder)
    result = evaluate(monitor, audited)
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest.get('driver_exit_code') not in (None, 0) and not manifest.get('termination'):
        manifest['error'] = f'driver exited {manifest["driver_exit_code"]}: ' + json.dumps(audited['diagnostics'])
        result['success'] = False
    manifest['model_metadata'] = model_metadata(folder)
    bag_metadata = folder/'bag/metadata.yaml'
    if bag_metadata.exists():
        metadata = yaml.safe_load(bag_metadata.read_text())['rosbag2_bagfile_information']
        counts = {topic['topic_metadata']['name']: topic['message_count']
                  for topic in metadata['topics_with_message_count']}
        manifest['bag_topic_counts'] = counts
        manifest['bag_complete'] = all(counts.get(topic, 0) > 0 for topic in TOPICS if topic != '/exploration_result')
        evidence = verify_bag(folder)
        manifest['bag_evidence_complete'] = (evidence['all_observation_frames_in_bag']
            and not evidence['movement_commands_missing_from_bag']
            and not evidence['movement_terminal_statuses_missing_from_bag'])
    if manifest.get('task_type') == 'explore-map' and manifest.get('mode') != 'tool_diagnostic':
        result = exploration_verdict(manifest, summarize(folder), benchmark_bag(folder), audited)
    save(folder/'manifest.json', manifest)
    save(folder/'evaluation.json', result)
    write_report(folder)


if __name__ == '__main__':
    main()
