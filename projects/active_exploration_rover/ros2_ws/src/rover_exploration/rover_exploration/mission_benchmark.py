"""
V16 multi-start mission benchmark runner.

Runs a fixed matrix of spawn poses sequentially, records each mission
with ros2 bag, evaluates the recording with the existing V15.3 mission
evaluator (read_bag + evaluate_mission, reused verbatim), and emits
aggregate JSON + Markdown reports.

The benchmark is evaluation infrastructure only. It never changes
planner, completion, recovery, SLAM, EKF, or controller behavior, and it
does not use ground truth for navigation or control.

Process ownership: the recorder and the launch process are started as
separate, independently owned process groups (start_new_session=True).
Cleanup signals only those two groups, in the order recorder-then-launch,
with bounded escalation. No pkill / killall / PPID scanning is used.

All subprocess, clock, sleep, signal, and completion-observation seams
are injectable so the logic is unit-tested without a live ROS/Gazebo
runtime.
"""

import argparse
import collections
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

from rover_exploration import mission_evaluator as me

import yaml


SCHEMA_VERSION = 1

# Canonical recorder topic set: the evidence required by the existing
# evaluator (me.REQUIRED_TOPICS) plus recovery / diagnostics topics.
RECORDER_TOPICS = [
    '/clock',
    '/exploration_complete',
    '/exploration_result',
    '/map',
    '/planning_grid',
    '/planned_path',
    '/cmd_vel',
    '/cmd_vel_raw',
    '/odometry/filtered',
    '/ground_truth/odometry',
    '/model/kd_bot/odometry',
    '/imu/data_raw',
    '/tf',
    '/tf_static',
    '/recovery_request',
    '/recovery_status',
    '/scan',
    '/rosout',
]

DEFAULT_CONFIG_RELPATH = os.path.join('config', 'mission_benchmark_v16.yaml')

# Top-level and nested config keys accepted by the validator. Any other
# key is rejected as an unknown configuration key.
_TOP_LEVEL_KEYS = {'schema_version', 'defaults', 'poses'}
_DEFAULT_KEYS = {'wall_timeout_s', 'post_completion_sim_s'}
_POSE_KEYS = {
    'name', 'spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw',
}

# Bounded cleanup windows (seconds).
_DEFAULT_FLUSH_TIMEOUT_S = 5.0
_DEFAULT_LAUNCH_SHUTDOWN_TIMEOUT_S = 30.0
# A simulation clock that makes no progress for this long (wall) while the
# launch is still alive indicates a hung simulation.
_CLOCK_STALL_WALL_S = 30.0


class ConfigError(Exception):
    """Raised when a benchmark configuration is invalid."""


def _is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(
        value, bool
    ) and math.isfinite(value)


def _safe_name(name):
    return isinstance(name, str) and len(name) > 0 and all(
        c.isalnum() or c == '_' for c in name
    )


def load_config(path):
    """
    Load and validate a benchmark YAML configuration.

    Returns the parsed dict. Raises ConfigError on any schema violation.
    """
    if not os.path.isfile(path):
        raise ConfigError(f'config file not found: {path}')
    try:
        with open(path, 'r') as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f'could not parse config {path}: {error}')

    if not isinstance(raw, dict):
        raise ConfigError('config root must be a mapping')

    unknown_top = set(raw.keys()) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ConfigError(
            'unknown config key(s): ' + ', '.join(sorted(unknown_top))
        )

    if raw.get('schema_version') != SCHEMA_VERSION:
        raise ConfigError(
            f'schema_version must be {SCHEMA_VERSION}, got '
            f'{raw.get("schema_version")!r}'
        )

    defaults = raw.get('defaults')
    if not isinstance(defaults, dict):
        raise ConfigError('defaults must be a mapping')
    unknown_def = set(defaults.keys()) - _DEFAULT_KEYS
    if unknown_def:
        raise ConfigError(
            'unknown defaults key(s): ' + ', '.join(sorted(unknown_def))
        )
    wall = defaults.get('wall_timeout_s')
    post = defaults.get('post_completion_sim_s')
    if not _is_finite(wall) or wall <= 0.0:
        raise ConfigError('defaults.wall_timeout_s must be finite and > 0')
    if not _is_finite(post) or post < 2.0:
        raise ConfigError(
            'defaults.post_completion_sim_s must be finite and >= 2.0'
        )

    poses = raw.get('poses')
    if not isinstance(poses, list) or not poses:
        raise ConfigError('poses must be a non-empty list')

    seen = set()
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict):
            raise ConfigError(f'pose[{index}] must be a mapping')
        unknown_pose = set(pose.keys()) - _POSE_KEYS
        if unknown_pose:
            raise ConfigError(
                f'pose[{index}] unknown key(s): '
                + ', '.join(sorted(unknown_pose))
            )
        name = pose.get('name')
        if not _safe_name(name):
            raise ConfigError(
                f'pose[{index}] name must be a non-empty '
                'alphanumeric/underscore string'
            )
        if name in seen:
            raise ConfigError(f'duplicate pose name: {name!r}')
        seen.add(name)
        for key in ('spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw'):
            if not _is_finite(pose.get(key)):
                raise ConfigError(f'pose {name!r} {key} must be finite')

    return raw


def resolve_default_config():
    """Find the installed benchmark config, falling back to the source tree."""
    try:
        from ament_index_python.packages import (
            get_package_share_directory,
        )
        share = get_package_share_directory('rover_exploration')
        installed = os.path.join(share, DEFAULT_CONFIG_RELPATH)
        if os.path.isfile(installed):
            return installed
    except Exception:
        pass
    local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        DEFAULT_CONFIG_RELPATH,
    )
    return local


def filter_poses(config, selected=None):
    """
    Return poses in config order, optionally restricted to `selected`.

    Raises ConfigError on an unknown or duplicate selected name.
    """
    poses = config['poses']
    if not selected:
        return list(poses)
    chosen = []
    seen = set()
    for name in selected:
        if name in seen:
            raise ConfigError(f'duplicate --runs entry: {name!r}')
        seen.add(name)
        match = None
        for pose in poses:
            if pose['name'] == name:
                match = pose
                break
        if match is None:
            raise ConfigError(f'unknown pose name: {name!r}')
        chosen.append(match)
    return chosen


def build_launch_command(pose, enable_rviz=False, enable_motion=True):
    """Construct the exact ros2 launch argv list for one pose."""
    return [
        'ros2', 'launch', 'rover_exploration', 'exploration.launch.py',
        f'enable_motion:={str(bool(enable_motion)).lower()}',
        f'enable_rviz:={str(bool(enable_rviz)).lower()}',
        f'spawn_x:={float(pose["spawn_x"])}',
        f'spawn_y:={float(pose["spawn_y"])}',
        f'spawn_z:={float(pose["spawn_z"])}',
        f'spawn_yaw:={float(pose["spawn_yaw"])}',
    ]


def build_recorder_command(bag_dir):
    """Construct the exact ros2 bag record argv list."""
    return [
        'ros2', 'bag', 'record',
        '-o', bag_dir,
        '--topics',
    ] + list(RECORDER_TOPICS)


class BenchmarkRunner:
    """
    Orchestrate a sequential multi-start mission benchmark.

    Subprocess / clock / sleep / signal seams are injectable for tests.
    """

    def __init__(
        self,
        config,
        output_dir,
        selected=None,
        wall_timeout_s=None,
        post_completion_s=None,
        execute=False,
        popen=None,
        now=time.monotonic,
        sleep=time.sleep,
        killpg=os.killpg,
        getpgid=os.getpgid,
        make_provider=None,
        flush_timeout_s=_DEFAULT_FLUSH_TIMEOUT_S,
        launch_shutdown_timeout_s=_DEFAULT_LAUNCH_SHUTDOWN_TIMEOUT_S,
        clock_stall_wall_s=_CLOCK_STALL_WALL_S,
        log=None,
    ):
        """Construct the runner; apply CLI overrides and injectable seams."""
        self.config = config
        self.output_dir = output_dir
        self.poses = filter_poses(config, selected)
        defaults = config.get('defaults', {})
        self.wall_timeout_s = (
            float(wall_timeout_s)
            if wall_timeout_s is not None
            else float(defaults['wall_timeout_s'])
        )
        self.post_completion_s = (
            float(post_completion_s)
            if post_completion_s is not None
            else float(defaults['post_completion_sim_s'])
        )
        self.execute = execute
        self._popen = popen if popen is not None else subprocess.Popen
        self._now = now
        self._sleep = sleep
        self._killpg = killpg
        self._getpgid = getpgid
        self._make_provider = make_provider
        self._flush_timeout_s = flush_timeout_s
        self._launch_shutdown_timeout_s = launch_shutdown_timeout_s
        self._clock_stall_wall_s = clock_stall_wall_s
        self._log = log if log is not None else print
        self.run_records = []

    # -- planning ---------------------------------------------------------

    def plan(self):
        """Build the full execution plan as a list of per-run dicts."""
        plan = []
        for pose in self.poses:
            bag_dir = os.path.join(
                self.output_dir, 'runs', pose['name']
            )
            launch_cmd = build_launch_command(pose, enable_rviz=False)
            recorder_cmd = build_recorder_command(bag_dir)
            plan.append({
                'pose_name': pose['name'],
                'spawn': {
                    'spawn_x': float(pose['spawn_x']),
                    'spawn_y': float(pose['spawn_y']),
                    'spawn_z': float(pose['spawn_z']),
                    'spawn_yaw': float(pose['spawn_yaw']),
                },
                'launch_args': launch_cmd,
                'recorder_args': recorder_cmd,
                'topics': list(RECORDER_TOPICS),
                'wall_timeout_s': self.wall_timeout_s,
                'post_completion_sim_s': self.post_completion_s,
                'bag_dir': bag_dir,
            })
        return plan

    def _print_plan(self):
        self._log('=== V16 mission benchmark plan (dry run) ===')
        self._log(f'output_dir: {self.output_dir}')
        self._log(f'wall_timeout_s: {self.wall_timeout_s}')
        self._log(
            f'post_completion_sim_s: {self.post_completion_s}'
        )
        for entry in self.plan():
            self._log(f'\n--- pose: {entry["pose_name"]} ---')
            self._log('launch: ' + ' '.join(entry['launch_args']))
            self._log('record: ' + ' '.join(entry['recorder_args']))
            self._log(f'topics ({len(entry["topics"])}): '
                      + ' '.join(entry['topics']))

    # -- execution --------------------------------------------------------

    def run(self):
        """Execute the benchmark (or print the plan in dry-run mode)."""
        if not self.execute:
            self._print_plan()
            os.makedirs(self.output_dir, exist_ok=True)
            with open(
                os.path.join(self.output_dir, 'plan.json'), 'w'
            ) as handle:
                json.dump(
                    {
                        'schema_version': SCHEMA_VERSION,
                        'dry_run': True,
                        'config': self.config,
                        'plan': self.plan(),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            return 0

        os.makedirs(os.path.join(self.output_dir, 'runs'), exist_ok=True)
        for pose in self.poses:
            record = self._run_pose(pose)
            self.run_records.append(record)
            status = record.get('status')
            self._log(
                f'pose {pose["name"]}: {status} '
                f'(pass={record.get("passed")})'
            )
            # Continue to the next pose even on failure, unless safe
            # cleanup could not be confirmed.
            if status == 'cleanup_failed':
                break

        self._write_reports()
        attempted = len(self.run_records)
        passed = sum(
            1 for r in self.run_records if r.get('passed')
        )
        failed = attempted - passed
        overall = failed == 0 and attempted > 0
        self._log(
            f'\n=== benchmark complete: '
            f'{passed}/{attempted} passed ==='
        )
        return 0 if overall else 1

    def _run_pose(self, pose):
        run_dir = os.path.join(self.output_dir, 'runs', pose['name'])
        bag_dir = os.path.join(run_dir, 'bag')
        recorder_log = os.path.join(run_dir, 'recorder.log')
        launch_log = os.path.join(run_dir, 'launch.log')
        os.makedirs(run_dir, exist_ok=True)

        launch_cmd = build_launch_command(pose, enable_rviz=False)
        recorder_cmd = build_recorder_command(bag_dir)

        manifest = {
            'pose_name': pose['name'],
            'spawn': {
                'spawn_x': float(pose['spawn_x']),
                'spawn_y': float(pose['spawn_y']),
                'spawn_z': float(pose['spawn_z']),
                'spawn_yaw': float(pose['spawn_yaw']),
            },
            'launch_args': launch_cmd,
            'recorder_args': recorder_cmd,
            'topics': list(RECORDER_TOPICS),
            'wall_timeout_s': self.wall_timeout_s,
            'post_completion_sim_s': self.post_completion_s,
            'start_time': self._now(),
        }
        with open(os.path.join(run_dir, 'manifest.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        recorder_proc = None
        launch_proc = None
        provider = None
        recorder_pgid = None
        launch_pgid = None
        record = {
            'pose_name': pose['name'],
            'spawn': manifest['spawn'],
            'launch_args': launch_cmd,
            'recorder_args': recorder_cmd,
            'status': 'unknown',
        }
        try:
            # Recorder starts BEFORE the launch so the initial
            # transient-local /exploration_complete=false state is
            # captured. The live subscriber is created between the
            # recorder and the launch for the same reason.
            with open(recorder_log, 'w') as rlog:
                recorder_proc = self._popen(
                    recorder_cmd,
                    stdout=rlog,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            recorder_pgid = self._getpgid(recorder_proc.pid)

            provider = None
            if self._make_provider is not None:
                provider = self._make_provider()

            with open(launch_log, 'w') as llog:
                launch_proc = self._popen(
                    launch_cmd,
                    stdout=llog,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            launch_pgid = self._getpgid(launch_proc.pid)

            status, completion_sim_s, error = self._watch(
                recorder_proc, launch_proc, provider
            )
            record['status'] = status
            if error:
                record['error'] = error
            record['completion_time_s'] = (
                completion_sim_s if completion_sim_s is not None else None
            )

            if status == 'completed' and completion_sim_s is not None:
                # Stop and flush the recorder FIRST.
                recorder_ok = self._stop_group(
                    recorder_proc, recorder_pgid, 'recorder',
                    self._flush_timeout_s,
                )
                # Then stop the launch tree.
                launch_ok = self._stop_group(
                    launch_proc, launch_pgid, 'launch',
                    self._launch_shutdown_timeout_s,
                )
                if recorder_ok and launch_ok:
                    self._evaluate(run_dir, bag_dir, record)
                else:
                    record['status'] = 'cleanup_failed'
                    record['error'] = (
                        'process group did not stop after completion'
                    )
                    record['passed'] = False
            else:
                # No valid completion: stop recorder first, then launch,
                # in the same owned-group order.
                recorder_ok = self._stop_group(
                    recorder_proc, recorder_pgid, 'recorder',
                    self._flush_timeout_s,
                )
                launch_ok = self._stop_group(
                    launch_proc, launch_pgid, 'launch',
                    self._launch_shutdown_timeout_s,
                )
                if not (recorder_ok and launch_ok):
                    record['status'] = 'cleanup_failed'
                    record['error'] = (
                        'process group did not stop after incomplete run'
                    )
                    record['passed'] = False

        except KeyboardInterrupt:
            record['status'] = 'interrupted'
            record['error'] = 'user interruption'
            self._cleanup(
                recorder_proc, recorder_pgid, launch_proc, launch_pgid
            )
            raise
        except Exception as error:  # noqa: BLE001 - report, keep going
            record['status'] = 'error'
            record['error'] = str(error)
            cleanup_ok = self._cleanup(
                recorder_proc, recorder_pgid, launch_proc, launch_pgid
            )
            if not cleanup_ok:
                # A failed cleanup is authoritative: the run cannot be
                # considered safe, so do not continue to the next pose.
                record['status'] = 'cleanup_failed'
                record['passed'] = False
                record['cleanup_error'] = (
                    'process group did not stop after exception'
                )
                record['original_error'] = str(error)
        finally:
            if provider is not None:
                try:
                    provider.shutdown()
                except Exception:
                    pass
            # If a partial startup left the groups alive, ensure cleanup.
            if (
                recorder_proc is not None
                and self._group_alive(recorder_pgid, recorder_proc)
            ) or (
                launch_proc is not None
                and self._group_alive(launch_pgid, launch_proc)
            ):
                self._cleanup(
                    recorder_proc, recorder_pgid,
                    launch_proc, launch_pgid,
                )

        return record

    def _watch(self, recorder_proc, launch_proc, provider):
        """
        Monitor /exploration_complete false->true, then wait post time.

        Returns (status, completion_sim_s, error). provider may be None
        for tests that drive completion through an injected fake.
        """
        start_wall = self._now()
        saw_false = False
        transition_sim = None
        clock_sim = 0.0
        last_clock_sim = None
        last_clock_wall = start_wall
        first_clock_wall = None

        # If no provider is injected, completion cannot be observed; this
        # only happens in unit tests that supply a provider.
        if provider is None:
            # Block until launch/recorder exit or wall timeout, so a test
            # without a provider does not spin forever.
            while True:
                if launch_proc.poll() is not None:
                    return 'launch_exit', None, 'launch exited'
                if recorder_proc.poll() is not None:
                    return 'recorder_exit', None, 'recorder exited'
                if self._now() - start_wall > self.wall_timeout_s:
                    return 'wall_timeout', None, 'wall timeout'
                self._sleep(0.2)

        while True:
            if launch_proc.poll() is not None:
                return 'launch_exit', None, 'launch process exited'
            if recorder_proc.poll() is not None:
                return 'recorder_exit', None, 'recorder process exited'
            if self._now() - start_wall > self.wall_timeout_s:
                return 'wall_timeout', None, 'wall-clock timeout'

            # Clock-start guard: if a provider exists but no /clock has
            # arrived within the stall window, the simulation clock never
            # started (e.g. Gazebo failed to publish).
            if provider is not None and first_clock_wall is None and (
                self._now() - start_wall > self._clock_stall_wall_s
            ):
                return (
                    'clock_not_started', None,
                    'simulation clock never started',
                )

            event = provider.poll(timeout=0.2)
            if event is not None:
                if event['type'] == 'clock':
                    clock_sim = event['sim']
                    if last_clock_sim is not None and (
                        clock_sim > last_clock_sim + 1e-9
                    ):
                        last_clock_wall = self._now()
                    if last_clock_sim is None:
                        # First clock message: anchor the stall window.
                        first_clock_wall = self._now()
                        last_clock_wall = first_clock_wall
                    last_clock_sim = clock_sim
                elif event['type'] == 'completion':
                    clock_sim = event['sim']
                    value = event['value']
                    if value is False:
                        saw_false = True
                    else:
                        if not saw_false:
                            return (
                                'no_false_state', None,
                                'completion true before any false',
                            )
                        if transition_sim is None:
                            transition_sim = clock_sim

            else:
                # No pending event: yield so the loop does not busy-spin
                # while waiting for the simulation to advance.
                self._sleep(0.05)

            # Post-completion observation window (simulation time).
            if transition_sim is not None and (
                clock_sim - transition_sim
            ) >= self.post_completion_s:
                return 'completed', transition_sim, None

            # Hung-simulation guard: clock stopped advancing.
            if (
                last_clock_sim is not None
                and (self._now() - last_clock_wall)
                > self._clock_stall_wall_s
            ):
                return (
                    'clock_stopped', None,
                    'simulation clock stopped advancing',
                )

    def _evaluate(self, run_dir, bag_dir, record):
        try:
            collected = me.read_bag(bag_dir)
            result, passed, reasons = me.evaluate_mission(collected)
        except Exception as error:  # noqa: BLE001 - evaluator/IO failure
            record['evaluation_error'] = str(error)
            record['status'] = 'eval_error'
            record['passed'] = False
            return
        result_path = os.path.join(run_dir, 'evaluator_result.json')
        with open(result_path, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        record['evaluator_result'] = result
        record['passed'] = bool(passed)
        record['failure_reasons'] = reasons
        if not passed:
            record['status'] = 'failed'
        else:
            record['status'] = 'passed'

    # -- cleanup ----------------------------------------------------------

    def _group_alive(self, pgid, proc):
        """
        Return True if the owned process group still exists.

        Reaps the group leader via proc.poll() so an exited leader is not
        mistaken for a live group, then probes the whole captured group with
        killpg(pgid, 0). ProcessLookupError (ESRCH) means the group is gone;
        PermissionError (EPERM) or any other probe error means it is still
        alive. Only the captured owned pgid is ever passed here.
        """
        if pgid is None:
            return False
        try:
            if proc is not None:
                proc.poll()  # reap the leader if it has exited
            self._killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # EPERM: not ours to probe, but definitely still present.
            return True
        except OSError:
            # Other probe errors (e.g. ESRCH already caught above): treat
            # as still alive rather than risk a false "gone".
            return True
        return True

    def _signal_group_by_pgid(self, pgid, signum):
        if pgid is None:
            return
        try:
            self._killpg(pgid, signum)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001 - best-effort
            pass

    def _wait_group_gone(self, pgid, proc, timeout_s):
        deadline = self._now() + timeout_s
        while self._now() < deadline:
            if not self._group_alive(pgid, proc):
                return True
            self._sleep(0.1)
        return not self._group_alive(pgid, proc)

    def _stop_group(self, proc, pgid, label, timeout_s):
        """
        Stop a group: SIGINT, wait for it to disappear, then escalate.

        Reaps the leader and checks the whole captured group each wait, so a
        reaped leader does not hide surviving descendants. Returns True only if
        the entire process group is confirmed gone. Signals only the captured
        owned pgid; never any other group.
        """
        self._signal_group_by_pgid(pgid, signal.SIGINT)
        if self._wait_group_gone(pgid, proc, timeout_s):
            return True
        self._signal_group_by_pgid(pgid, signal.SIGTERM)
        if self._wait_group_gone(pgid, proc, timeout_s):
            return True
        self._signal_group_by_pgid(pgid, signal.SIGKILL)
        # Bounded final wait after SIGKILL.
        return self._wait_group_gone(pgid, proc, timeout_s)

    def _cleanup(self, recorder_proc, recorder_pgid, launch_proc, launch_pgid):
        """
        Shut down recorder then launch, owned groups only.

        Returns True only if both owned process groups are confirmed gone.
        """
        ok = True
        if not self._stop_group(
            recorder_proc, recorder_pgid, 'recorder', self._flush_timeout_s
        ):
            ok = False
        if not self._stop_group(
            launch_proc, launch_pgid, 'launch',
            self._launch_shutdown_timeout_s,
        ):
            ok = False
        return ok

    # -- reports ----------------------------------------------------------

    def _write_reports(self):
        aggregate = self._aggregate()
        summary_path = os.path.join(self.output_dir, 'benchmark_summary.json')
        with open(summary_path, 'w') as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True)
        md_path = os.path.join(self.output_dir, 'benchmark_summary.md')
        with open(md_path, 'w') as handle:
            handle.write(self._markdown(aggregate))
        return aggregate

    def _aggregate(self):
        records = self.run_records
        attempted = len(records)
        evaluated = [
            r for r in records if r.get('evaluator_result') is not None
        ]
        passed = sum(1 for r in records if r.get('passed'))
        failed = attempted - passed

        completion_times = [
            r['completion_time_s']
            for r in evaluated
            if r.get('completion_time_s') is not None
        ]
        coverages = [
            r['evaluator_result'].get('known_map_percent', float('nan'))
            for r in evaluated
        ]
        recovery_counts = [
            r['evaluator_result'].get('recovery_requests', 0)
            for r in evaluated
        ]
        pos_errors = [
            r['evaluator_result'].get(
                'maximum_filtered_position_error_m', 0.0
            )
            for r in evaluated
        ]
        yaw_errors = [
            r['evaluator_result'].get('maximum_filtered_yaw_error_deg', 0.0)
            for r in evaluated
        ]
        trans_steps = [
            r['evaluator_result'].get(
                'maximum_map_to_odom_translation_step_m', 0.0
            )
            for r in evaluated
        ]
        yaw_steps = [
            r['evaluator_result'].get(
                'maximum_map_to_odom_yaw_step_deg', 0.0
            )
            for r in evaluated
        ]

        def _mean(values):
            return (sum(values) / len(values)) if values else None

        aggregate = {
            'schema_version': SCHEMA_VERSION,
            'config_used': {
                'wall_timeout_s': self.wall_timeout_s,
                'post_completion_sim_s': self.post_completion_s,
                'poses': [
                    {
                        'name': p['name'],
                        'spawn_x': float(p['spawn_x']),
                        'spawn_y': float(p['spawn_y']),
                        'spawn_z': float(p['spawn_z']),
                        'spawn_yaw': float(p['spawn_yaw']),
                    }
                    for p in self.poses
                ],
            },
            'selected_poses': [
                {
                    'name': r['pose_name'],
                    'spawn': r.get('spawn'),
                }
                for r in records
            ],
            'overall_pass': failed == 0 and attempted > 0,
            'runs_attempted': attempted,
            'runs_passed': passed,
            'runs_failed': failed,
            'pass_rate': (
                (passed / attempted) if attempted else 0.0
            ),
            'evaluated_run_count': len(evaluated),
            'per_run': [],
            'aggregate_metrics': {
                'mean_completion_time_s': _mean(completion_times),
                'max_completion_time_s': (
                    max(completion_times) if completion_times else None
                ),
                'min_coverage_percent': (
                    min(coverages) if coverages else None
                ),
                'total_recovery_requests': sum(recovery_counts),
                'max_filtered_position_error_m': (
                    max(pos_errors) if pos_errors else None
                ),
                'max_filtered_yaw_error_deg': (
                    max(yaw_errors) if yaw_errors else None
                ),
                'max_map_to_odom_translation_step_m': (
                    max(trans_steps) if trans_steps else None
                ),
                'max_map_to_odom_yaw_step_deg': (
                    max(yaw_steps) if yaw_steps else None
                ),
            },
        }

        for r in records:
            result = r.get('evaluator_result') or {}
            per = {
                'pose_name': r['pose_name'],
                'status': r.get('status'),
                'passed': r.get('passed'),
                'launch_args': r.get('launch_args'),
                'recorder_args': r.get('recorder_args'),
                'completion_time_s': r.get('completion_time_s'),
                'lifecycle_error': (
                    r.get('error') or r.get('evaluation_error')
                ),
                'evaluator_result': result,
                'coverage_percent': result.get('known_map_percent'),
                'goals_assigned': result.get('goals_assigned'),
                'goals_reached': result.get('goals_reached'),
                'temporary_failures': result.get(
                    'temporary_failure_events'
                ),
                'permanent_failures': result.get(
                    'permanent_failed_regions'
                ),
                'recovery_requests': result.get('recovery_requests'),
                'max_filtered_position_error_m': result.get(
                    'maximum_filtered_position_error_m'
                ),
                'max_filtered_yaw_error_deg': result.get(
                    'maximum_filtered_yaw_error_deg'
                ),
                'max_map_to_odom_translation_step_m': result.get(
                    'maximum_map_to_odom_translation_step_m'
                ),
                'max_map_to_odom_yaw_step_deg': result.get(
                    'maximum_map_to_odom_yaw_step_deg'
                ),
                'post_completion_observation_s': result.get(
                    'post_completion_observation_s'
                ),
                'active_cmd_vel_after_completion': result.get(
                    'active_cmd_vel_after_completion'
                ),
                'active_cmd_vel_raw_after_completion': result.get(
                    'active_cmd_vel_raw_after_completion'
                ),
                'nonempty_paths_after_completion': result.get(
                    'nonempty_paths_after_completion'
                ),
                'ground_truth_motion_after_completion_m': result.get(
                    'ground_truth_motion_after_completion_m'
                ),
                'failure_reasons': result.get('failure_reasons'),
                'error': r.get('error') or r.get('evaluation_error'),
            }
            aggregate['per_run'].append(per)

        return aggregate

    def _markdown(self, aggregate):
        lines = []
        lines.append('# V16 Mission Benchmark Summary')
        lines.append('')
        lines.append(
            f"- Overall pass: **{aggregate['overall_pass']}**"
        )
        lines.append(
            f"- Runs attempted: {aggregate['runs_attempted']}"
        )
        lines.append(f"- Runs passed: {aggregate['runs_passed']}")
        lines.append(f"- Runs failed: {aggregate['runs_failed']}")
        lines.append(
            f"- Pass rate: {aggregate['pass_rate']:.3f}"
        )
        lines.append(
            f"- Evaluated runs: {aggregate['evaluated_run_count']}"
        )
        lines.append('')
        lines.append('## Per-pose results')
        lines.append('')
        lines.append('| Pose | Status | Pass | Completion (s) | '
                     'Coverage % | Goals A/R | Recovery |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- |')
        for per in aggregate['per_run']:
            completion = per['completion_time_s']
            completion = (
                f'{completion:.2f}' if completion is not None else '-'
            )
            coverage = per['coverage_percent']
            coverage = (
                f'{coverage:.2f}' if coverage is not None else '-'
            )
            goals = (
                f"{per['goals_assigned']}/{per['goals_reached']}"
                if per['goals_assigned'] is not None else '-'
            )
            recovery = (
                str(per['recovery_requests'])
                if per['recovery_requests'] is not None else '-'
            )
            lines.append(
                f'| {per["pose_name"]} | {per["status"]} | '
                f'{per["passed"]} | {completion} | {coverage} | '
                f'{goals} | {recovery} |'
            )
        lines.append('')
        lines.append('## Failure reasons')
        lines.append('')
        any_reason = False
        for per in aggregate['per_run']:
            reasons = per.get('failure_reasons')
            if reasons:
                any_reason = True
                name = per['pose_name']
                lines.append(f'### {name}')
                for reason in reasons:
                    lines.append(f'- {reason}')
        if not any_reason:
            lines.append('None.')
        lines.append('')
        lines.append('## Lifecycle and process status')
        lines.append('')
        any_lifecycle = False
        for per in aggregate['per_run']:
            status = per.get('status')
            lifecycle_error = per.get('lifecycle_error')
            if status not in ('passed', 'failed') or lifecycle_error:
                any_lifecycle = True
                name = per['pose_name']
                lines.append(f'### {name}')
                lines.append(f'- status: {status}')
                if lifecycle_error:
                    lines.append(f'- error: {lifecycle_error}')
                else:
                    lines.append(
                        '- error: none (evaluator result present)'
                    )
        if not any_lifecycle:
            lines.append(
                'All runs reached the evaluator with no lifecycle or '
                'process errors.'
            )
        lines.append('')
        lines.append('## Aggregate metrics')
        lines.append('')
        am = aggregate['aggregate_metrics']
        for key, value in am.items():
            lines.append(f'- {key}: {value}')
        lines.append('')
        return '\n'.join(lines)


class LiveCompletionProvider:
    """
    Real rclpy observer for /exploration_complete and /clock.

    Uses a dedicated rclpy Context so sequential missions do not leak
    global rclpy state. Subscribes to /exploration_complete with
    reliable + transient-local + depth-1 QoS (so the initial latched
    false is observed), and to /clock with a depth-1 QoS. High-rate
    /clock messages are collapsed: the provider tracks only the latest
    simulation time and emits at most one clock event per advancement,
    so no unbounded queue accumulates. Completion order is preserved.
    It never reports true until this run has observed a false.
    """

    def __init__(self, clock_start_timeout_s=_CLOCK_STALL_WALL_S):
        import rclpy
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import Bool
        from rosgraph_msgs.msg import Clock

        self._rclpy = rclpy
        self._Clock = Clock
        self._Bool = Bool
        self._clock_start_timeout_s = clock_start_timeout_s

        self._context = rclpy.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(
            'v16_completion_watcher', context=self._context
        )

        completion_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        clock_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._lock = threading.Lock()
        self._latest_clock_sim = 0.0
        self._clock_started = False
        self._last_emitted_clock_sim = None
        self._saw_false = False
        self._queue = collections.deque(maxlen=64)

        self._node.create_subscription(
            self._Bool,
            '/exploration_complete',
            self._on_completion,
            completion_qos,
        )
        self._node.create_subscription(
            self._Clock,
            '/clock',
            self._on_clock,
            clock_qos,
        )

        self._executor = rclpy.executors.SingleThreadedExecutor(
            context=self._context
        )
        self._executor.add_node(self._node)
        self._stop = False
        self._thread = threading.Thread(
            target=self._spin, name='v16-completion-watcher', daemon=True
        )
        self._thread.start()

    def _on_clock(self, msg):
        sim = float(msg.clock.sec) + float(msg.clock.nanosec) / 1e9
        with self._lock:
            self._latest_clock_sim = sim
            self._clock_started = True

    def _on_completion(self, msg):
        with self._lock:
            value = bool(msg.data)
            if not value:
                self._saw_false = True
            sim = self._latest_clock_sim
            self._queue.append(
                {'type': 'completion', 'sim': sim, 'value': value}
            )

    def _spin(self):
        try:
            while not self._stop:
                self._executor.spin_once(timeout_sec=0.1)
        except Exception:
            pass

    def poll(self, timeout=0.2):
        """Return the next event, or None if none within `timeout`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                # A queued completion transition must be returned before a
                # newly advanced /clock event, otherwise the high-rate clock
                # could starve the completion signal indefinitely.
                if self._queue:
                    return self._queue.popleft()
                # Emit at most one clock event per advancement.
                if self._clock_started and (
                    self._last_emitted_clock_sim is None
                    or self._latest_clock_sim
                    > self._last_emitted_clock_sim + 1e-9
                ):
                    self._last_emitted_clock_sim = self._latest_clock_sim
                    return {'type': 'clock', 'sim': self._latest_clock_sim}
            time.sleep(0.02)
        return None

    def clock_started(self):
        """Return True once a /clock message has been received."""
        with self._lock:
            return self._clock_started

    def latest_clock_sim(self):
        with self._lock:
            return self._latest_clock_sim

    def saw_false(self):
        with self._lock:
            return self._saw_false

    def shutdown(self):
        self._stop = True
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            self._rclpy.shutdown(context=self._context)
        except Exception:
            pass


class FakeCompletionStream:
    """
    Injectable completion/clock event source for tests.

    events: list of dicts, each {'type':'completion'|'clock', 'sim':float,
    'value':bool (for completion)}.
    """

    def __init__(self, events):
        """Store the scripted event list (completion/clock dicts)."""
        self._events = list(events)
        self._index = 0

    def poll(self, timeout=0.2):
        """Return the next event, or None when the script is exhausted."""
        if self._index >= len(self._events):
            return None
        event = self._events[self._index]
        self._index += 1
        return event

    def shutdown(self):
        """Release any provider resources (no-op for the fake)."""
        pass


def main(args=None):
    """CLI entry point: parse args, run the benchmark, return exit code."""
    parser = argparse.ArgumentParser(
        description='Run the V16 multi-start mission benchmark.'
    )
    parser.add_argument('--config', default=None, help='path to YAML config')
    parser.add_argument(
        '--output-dir', required=True, help='benchmark output directory'
    )
    parser.add_argument(
        '--runs', nargs='+', default=None,
        help='optional subset of pose names to run',
    )
    parser.add_argument(
        '--wall-timeout-s', type=float, default=None,
        help='per-mission wall-clock timeout (seconds)',
    )
    parser.add_argument(
        '--post-completion-s', type=float, default=None,
        help='simulated seconds to record after completion',
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='run missions; without it, only print the plan',
    )
    parsed = parser.parse_args(args)

    # CLI override validation.
    if parsed.wall_timeout_s is not None and (
        not _is_finite(parsed.wall_timeout_s)
        or parsed.wall_timeout_s <= 0.0
    ):
        print('ERROR: --wall-timeout-s must be finite and > 0',
              file=sys.stderr)
        return 2
    if parsed.post_completion_s is not None and (
        not _is_finite(parsed.post_completion_s)
        or parsed.post_completion_s < 2.0
    ):
        print('ERROR: --post-completion-s must be finite and >= 2.0',
              file=sys.stderr)
        return 2

    config_path = parsed.config or resolve_default_config()
    try:
        config = load_config(config_path)
    except ConfigError as error:
        print(f'ERROR: invalid configuration: {error}', file=sys.stderr)
        return 2
    try:
        filter_poses(config, parsed.runs)
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2

    output_dir = parsed.output_dir
    # Never silently overwrite an existing benchmark.
    if os.path.isfile(
        os.path.join(output_dir, 'benchmark_summary.json')
    ):
        print(
            'ERROR: output directory already contains benchmark results; '
            'refusing to overwrite',
            file=sys.stderr,
        )
        return 2
    # Live execution refuses any non-empty target directory from a previous
    # live or partial benchmark (bags, manifests, logs, partial reports).
    # Dry-run is plan-only and may reuse an empty (or plan-only) directory.
    if parsed.execute and os.path.isdir(output_dir):
        try:
            entries = os.listdir(output_dir)
        except OSError:
            entries = []
        if entries:
            print(
                'ERROR: output directory already exists and is non-empty; '
                'refusing live execution to avoid overwriting prior data',
                file=sys.stderr,
            )
            return 2

    runner = BenchmarkRunner(
        config=config,
        output_dir=output_dir,
        selected=parsed.runs,
        wall_timeout_s=parsed.wall_timeout_s,
        post_completion_s=parsed.post_completion_s,
        execute=parsed.execute,
        make_provider=LiveCompletionProvider if parsed.execute else None,
    )

    try:
        return runner.run()
    except KeyboardInterrupt:
        print('ERROR: interrupted', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
