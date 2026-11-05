#!/usr/bin/env python3
"""Bounded isolated Gazebo CPU software-rendered physics/lidar smoke run.
Run after sourcing /opt/ros/jazzy/setup.bash. No global process termination.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import uuid


def messages(path):
    text = path.read_text()
    decoder = json.JSONDecoder()
    result = []
    while text.strip():
        text = text.lstrip()
        try:
            item, end = decoder.raw_decode(text)
        except json.JSONDecodeError:
            break
        result.append(item)
        text = text[end:]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(GZ_PARTITION='site_smoke_' + uuid.uuid4().hex,
               ROS_DOMAIN_ID='83', GZ_LOG_PATH=str(out / 'gz_logs'),
               ROS_LOG_DIR=str(out / 'ros_logs'), LIBGL_ALWAYS_SOFTWARE='1',
               MESA_LOADER_DRIVER_OVERRIDE='llvmpipe', EGL_PLATFORM='surfaceless')
    world = Path(__file__).with_name('smoke.sdf').resolve()
    specs = {
        'server': ['gz', 'sim', '-s', '-r', '--headless-rendering', '-v', '4',
                   '--iterations', '15000', str(world)],
        'scan': ['gz', 'topic', '-e', '-t', '/site_smoke/scan', '--json-output', '-n', '3'],
        'stats': ['gz', 'topic', '-e', '-t', '/world/site_localization_smoke/stats', '--json-output', '-n', '3'],
        'poses': ['gz', 'topic', '-e', '-t', '/world/site_localization_smoke/dynamic_pose/info', '--json-output', '-n', '3'],
    }
    children = {}
    handles = []
    started = time.monotonic()
    stop_reason = 'wall_guard'
    try:
        for key, cmd in specs.items():
            stdout = (out / (key + '.stdout')).open('w')
            stderr = (out / (key + '.stderr')).open('w')
            handles.extend([stdout, stderr])
            children[key] = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        deadline = started + 35
        while time.monotonic() < deadline:
            if all(children[key].poll() is not None for key in ('scan', 'stats', 'poses')):
                stop_reason = 'collectors_completed'
                break
            if children['server'].poll() is not None:
                stop_reason = 'server_exited'
                break
            time.sleep(0.2)
    finally:
        for proc in children.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for proc in children.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for handle in handles:
            handle.close()
    scan = messages(out / 'scan.stdout')
    stats = messages(out / 'stats.stdout')
    poses = messages(out / 'poses.stdout')
    summary = {
        'partition': env['GZ_PARTITION'], 'ros_domain_id': env['ROS_DOMAIN_ID'],
        'software_renderer_requested': True, 'world': str(world), 'commands': specs,
        'wall_seconds': time.monotonic() - started, 'stop_reason': stop_reason,
        'exit_codes': {key: p.returncode for key,p in children.items()},
        'messages': {'scan': len(scan), 'stats': len(stats), 'poses': len(poses)},
        'children_max_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'children_user_seconds': resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        'children_system_seconds': resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        'all_owned_processes_reaped': all(p.poll() is not None for p in children.values()),
        'scan_samples': scan[:1], 'stats_samples': stats, 'pose_samples': poses,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k,v in summary.items() if k not in ('scan_samples','stats_samples','pose_samples','commands')}, indent=2))
    return 0 if scan and len(stats) >= 2 and poses else 1

if __name__ == '__main__':
    raise SystemExit(main())
