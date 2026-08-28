"""
V16 mission benchmark runner tests.

No Gazebo or ROS runtime is launched. Subprocess creation, signals,
clocks, and completion observation are injected via fakes so the
process lifecycle, completion monitoring, cleanup ordering, and report
aggregation are exercised deterministically.
"""

import collections
import importlib.util
import os
import signal

import pytest

from rover_exploration import mission_benchmark as mb


HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)  # .../src/rover_exploration
SRC_DIR = os.path.dirname(PKG_DIR)  # .../src
CFG = os.path.join(PKG_DIR, 'config', 'mission_benchmark_v16.yaml')


VALID_CONFIG = {
    'schema_version': 1,
    'defaults': {'wall_timeout_s': 600, 'post_completion_sim_s': 5.0},
    'poses': [
        {'name': 'nominal', 'spawn_x': 0.0, 'spawn_y': 0.0,
         'spawn_z': 0.02, 'spawn_yaw': 0.0},
        {'name': 'translated_south', 'spawn_x': 0.0, 'spawn_y': -2.0,
         'spawn_z': 0.02, 'spawn_yaw': 0.0},
    ],
}


# --------------------------------------------------------------------------
# Fake subprocess / os seams
# --------------------------------------------------------------------------

class FakeProc:
    def __init__(self, pid, exit_after=None):
        self.pid = pid
        self._exit_after = exit_after
        self._leader_rc = None
        self._group_dead = False
        self._polled = False
        self._killed = []

    def poll(self):
        # Mark that the parent has reaped the leader. Callers (the runner's
        # group-liveness probe) use this to reflect a drained zombie.
        self._polled = True
        return self._leader_rc

    def wait(self, timeout=None):
        return self._leader_rc


class PopenHarness:
    """
    Record Popen calls and return FakeProcs with owned pgids.

    By default a SIGINT/SIGTERM also kills the whole group (fast
    cleanup in unit tests). Pass kill_group_on_kill=False to model the
    case where the group leader exits but child processes remain until
    SIGKILL -- used to verify full-group liveness checks.
    """

    def __init__(self, exit_codes=None, exit_first=False,
                 kill_group_on_kill=True):
        self.calls = []
        self.signal_calls = []
        self.procs = []
        self._next_pid = 100
        self.exit_codes = exit_codes or {}
        self.exit_first = exit_first
        self.kill_group_on_kill = kill_group_on_kill
        self._counter = 0

    def __call__(self, args, **kwargs):
        pid = self._next_pid
        self._next_pid += 1
        # pgid equals pid (start_new_session=True).
        proc = FakeProc(pid)
        proc.pgid = pid
        if self.exit_first and self._counter == 0:
            # Simulate the recorder exiting immediately (e.g. crash).
            proc._leader_rc = 1
        self.calls.append((args, kwargs))
        self.procs.append(proc)
        self._counter += 1
        return proc

    def getpgid(self, pid):
        return pid

    def killpg(self, pgid, signum):
        # Record the signal. Resolve the owned fake process.
        self.signal_calls.append((pgid, signum))
        for proc in self.procs:
            if proc.pgid == pgid:
                if signum == 0:
                    # Liveness probe: raise if the group is gone.
                    if proc._group_dead:
                        raise ProcessLookupError(
                            'process group %d does not exist' % pgid
                        )
                    return
                if signum == signal.SIGKILL:
                    # SIGKILL always removes the whole group.
                    proc._group_dead = True
                    proc._leader_rc = -signum
                else:
                    # SIGINT/SIGTERM: leader exits; whether the whole
                    # group dies depends on the harness flag (models
                    # "leader exits but children remain" when False).
                    proc._leader_rc = -signum
                    if self.kill_group_on_kill:
                        proc._group_dead = True
                return
        # Unknown pgid: never signal an unowned group.
        raise ProcessLookupError('unknown pgid %d' % pgid)


class KillpgRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, pgid, signum):
        self.calls.append((pgid, signum))

    def getpgid(self, pid):
        return pid


class ReapingHarness(PopenHarness):
    """
    Model a reaped leader whose group is gone once poll() reaps it.

    Before the leader is reaped (poll not yet called) the liveness probe
    still reports the group present -- exactly the zombie state that the
    V16.1 bug turned into a false cleanup_failed. After poll() drains the
    leader, killpg(pgid, 0) reports the group gone (no descendants).
    """

    def killpg(self, pgid, signum):
        self.signal_calls.append((pgid, signum))
        for proc in self.procs:
            if proc.pgid == pgid:
                if signum == 0:
                    if proc._polled:
                        raise ProcessLookupError('reaped')
                    return
                proc._leader_rc = -signum if signum else proc._leader_rc
                proc._group_dead = True
                return
        raise ProcessLookupError('unknown pgid %d' % pgid)


class DescendantHarness(PopenHarness):
    """
    Leader may be reaped, but a descendant keeps the group alive.

    Used to verify we keep signaling the group (not just the leader) and
    escalate to SIGKILL.
    """

    def killpg(self, pgid, signum):
        self.signal_calls.append((pgid, signum))
        for proc in self.procs:
            if proc.pgid == pgid:
                if signum == 0:
                    if proc._group_dead:
                        raise ProcessLookupError('gone')
                    return
                proc._leader_rc = -signum if signum else proc._leader_rc
                if signum == signal.SIGKILL:
                    proc._group_dead = True
                return
        raise ProcessLookupError('unknown pgid %d' % pgid)


class NeverDiesHarness(PopenHarness):
    """killpg(0) always reports the group present; cleanup must fail."""

    def killpg(self, pgid, signum):
        self.signal_calls.append((pgid, signum))
        for proc in self.procs:
            if proc.pgid == pgid:
                if signum == 0:
                    return  # group always present
                proc._leader_rc = -signum if signum else proc._leader_rc
                return
        raise ProcessLookupError('unknown pgid %d' % pgid)


def _fake_now(start=1000.0, step=0.0):
    state = {'t': start}

    def now():
        t = state['t']
        state['t'] += step
        return t

    return now


def _patch_evaluator(monkeypatch, results):
    """
    Patch me.read_bag / me.evaluate_mission to return canned results.

    results: list of (result_dict, passed) consumed per evaluate call.
    """
    iter_results = iter(results)

    def fake_read(bag_dir):
        return {}

    def fake_eval(collected):
        result, passed = next(iter_results)
        return result, passed, (
            result.get('failure_reasons', []) if not passed else []
        )

    monkeypatch.setattr(mb.me, 'read_bag', fake_read)
    monkeypatch.setattr(mb.me, 'evaluate_mission', fake_eval)


PASS_RESULT = {
    'schema_version': 1,
    'passed': True,
    'failure_reasons': [],
    'completion_time_s': 120.0,
    'post_completion_observation_s': 5.0,
    'known_map_percent': 98.5,
    'goals_assigned': 4,
    'goals_reached': 4,
    'temporary_failure_events': 0,
    'permanent_failed_regions': 0,
    'recovery_requests': 0,
    'maximum_filtered_position_error_m': 0.05,
    'maximum_filtered_yaw_error_deg': 0.4,
    'maximum_map_to_odom_translation_step_m': 0.10,
    'maximum_map_to_odom_yaw_step_deg': 0.2,
    'ground_truth_motion_after_completion_m': 0.0,
}


# --------------------------------------------------------------------------
# 1-6. Configuration parsing / rejection
# --------------------------------------------------------------------------

def test_valid_config_parses():
    cfg = mb.load_config(CFG)
    assert cfg['schema_version'] == 1
    assert len(cfg['poses']) == 4
    names = [p['name'] for p in cfg['poses']]
    assert names == [
        'nominal',
        'translated_south',
        'translated_north',
        'translated_south_yaw90',
    ]


def test_wrong_schema_version_rejected():
    bad = dict(VALID_CONFIG)
    bad['schema_version'] = 2
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'bad.yaml')
    with open(path, 'w') as handle:
        import yaml
        yaml.safe_dump(bad, handle)
    with pytest.raises(mb.ConfigError):
        mb.load_config(path)


def test_duplicate_or_unsafe_name_rejected():
    bad = dict(VALID_CONFIG)
    bad['poses'] = [
        {'name': 'ok_pose', 'spawn_x': 0.0, 'spawn_y': 0.0,
         'spawn_z': 0.02, 'spawn_yaw': 0.0},
        {'name': 'ok_pose', 'spawn_x': 1.0, 'spawn_y': 1.0,
         'spawn_z': 0.02, 'spawn_yaw': 0.0},
    ]
    with pytest.raises(mb.ConfigError):
        mb.load_config(_dump(bad))

    bad2 = dict(VALID_CONFIG)
    bad2['poses'] = [
        {'name': 'bad/name', 'spawn_x': 0.0, 'spawn_y': 0.0,
         'spawn_z': 0.02, 'spawn_yaw': 0.0},
    ]
    with pytest.raises(mb.ConfigError):
        mb.load_config(_dump(bad2))


def test_nonfinite_pose_rejected():
    bad = dict(VALID_CONFIG)
    bad['poses'] = [
        {'name': 'p', 'spawn_x': float('nan'), 'spawn_y': 0.0,
         'spawn_z': 0.02, 'spawn_yaw': 0.0},
    ]
    with pytest.raises(mb.ConfigError):
        mb.load_config(_dump(bad))


def test_invalid_timeouts_rejected():
    bad = dict(VALID_CONFIG)
    bad['defaults'] = {'wall_timeout_s': -1, 'post_completion_sim_s': 5.0}
    with pytest.raises(mb.ConfigError):
        mb.load_config(_dump(bad))
    bad['defaults'] = {'wall_timeout_s': 600, 'post_completion_sim_s': 1.0}
    with pytest.raises(mb.ConfigError):
        mb.load_config(_dump(bad))


def test_unknown_config_key_rejected():
    bad = dict(VALID_CONFIG)
    bad['extra_key'] = True
    with pytest.raises(mb.ConfigError):
        mb.load_config(_dump(bad))


# --------------------------------------------------------------------------
# 7-8. Launch / recorder argument construction
# --------------------------------------------------------------------------

def test_launch_argument_construction():
    pose = VALID_CONFIG['poses'][1]
    cmd = mb.build_launch_command(pose, enable_rviz=False)
    assert cmd[0:3] == ['ros2', 'launch', 'rover_exploration']
    assert 'exploration.launch.py' in cmd
    assert 'enable_motion:=true' in cmd
    assert 'enable_rviz:=false' in cmd
    assert 'spawn_x:=0.0' in cmd
    assert 'spawn_y:=-2.0' in cmd
    assert 'spawn_z:=0.02' in cmd
    assert 'spawn_yaw:=0.0' in cmd
    # Spawn coords forwarded unchanged (pose[1] = translated_south).
    assert 'spawn_x:=0.0' in cmd
    assert 'spawn_y:=-2.0' in cmd
    assert 'spawn_z:=0.02' in cmd
    assert 'spawn_yaw:=0.0' in cmd


def test_recorder_topic_construction():
    cmd = mb.build_recorder_command('/tmp/bag')
    assert cmd[0:3] == ['ros2', 'bag', 'record']
    assert cmd[3] == '-o'
    assert cmd[4] == '/tmp/bag'
    assert cmd[5] == '--topics'
    for topic in (
        '/clock', '/exploration_complete', '/exploration_result',
        '/map', '/tf', '/recovery_request', '/recovery_status',
        '/ground_truth/odometry', '/rosout',
    ):
        assert topic in cmd


# --------------------------------------------------------------------------
# 9-10. Dry run, recorder-before-launch
# --------------------------------------------------------------------------

def test_dry_run_starts_no_subprocesses():
    popen = PopenHarness()
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_dry',
        execute=False,
        popen=popen,
    )
    code = runner.run()
    assert code == 0
    assert popen.calls == []


def test_recorder_starts_before_launch():
    popen = PopenHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'clock', 'sim': 130.0},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_order',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(events),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    # Drive a passing evaluation without a real bag.
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission

    def fake_read(bag_dir):
        return {}

    def fake_eval(collected):
        return dict(PASS_RESULT), True, []

    mb.me.read_bag = fake_read
    mb.me.evaluate_mission = fake_eval
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    # Recorder must be the first process started.
    assert popen.calls[0][0][0:3] == ['ros2', 'bag', 'record']
    assert popen.calls[1][0][0:2] == ['ros2', 'launch']


# --------------------------------------------------------------------------
# 11-12. Completion requires false before true; post-completion sim time
# --------------------------------------------------------------------------

def test_completion_requires_false_before_true():
    popen = PopenHarness()
    # True arrives with no preceding false.
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 120.0, 'value': True},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_nofalse',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(events),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    assert runner.run_records[0]['status'] == 'no_false_state'


def test_post_completion_wait_uses_sim_time():
    popen = PopenHarness()
    # Transition at sim 100; need 5 sim-seconds more before stop order.
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 100.0, 'value': True},
        {'type': 'clock', 'sim': 103.0},  # not enough yet
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_post',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(events),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    # Still observing (clock only at 103 < 105): should not yet be
    # 'completed'. The provider is exhausted so it times out instead.
    assert runner.run_records[0]['status'] != 'completed'


# --------------------------------------------------------------------------
# 13-14. Timeout cleanup order; KeyboardInterrupt cleanup
# --------------------------------------------------------------------------

def test_timeout_cleanup_order_recorder_then_launch():
    popen = PopenHarness()
    # Never completes; provider exhausted -> wall/stall path returns.
    events = [{'type': 'clock', 'sim': 0.0}]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_timeout',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(events),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    runner.run()
    owned = {p.pgid for p in popen.procs}
    # Both groups signaled; first signal must be the recorder (pid 100).
    pgids = [c[0] for c in popen.signal_calls if c[1] == signal.SIGINT]
    assert pgids[0] == 100  # recorder group
    assert pgids[1] == 101  # launch group
    # Every signal targets an owned group only.
    assert all(p in owned for p in pgids)


def test_keyboard_interrupt_cleans_up_owned_groups():
    popen = PopenHarness()

    class BoomProvider:
        def poll(self, timeout=0.2):
            raise KeyboardInterrupt()

        def shutdown(self):
            pass

    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_kb',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: BoomProvider(),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    pgids = [c[0] for c in popen.signal_calls]
    assert 100 in pgids and 101 in pgids
    owned = {p.pgid for p in popen.procs}
    assert all(p in owned for p in pgids)


# --------------------------------------------------------------------------
# 15. Unexpected recorder/launch exit reported
# --------------------------------------------------------------------------

def test_unexpected_recorder_exit_reported():
    popen = PopenHarness(exit_first=True)
    events = [{'type': 'clock', 'sim': 0.0}]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_recdie',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(events),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    # exit_first=True makes the recorder (first spawned process) exit
    # immediately, simulating an unexpected recorder crash.
    runner.run()
    assert runner.run_records[0]['status'] == 'recorder_exit'


# --------------------------------------------------------------------------
# 16. Cleanup signals only owned process groups
# --------------------------------------------------------------------------

def test_cleanup_signals_only_owned_groups():
    popen = PopenHarness()
    events = [{'type': 'clock', 'sim': 0.0}]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_owned',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(events),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    runner.run()
    # Every killpg target must equal an owned pgid (recorder+launch per
    # pose). No pkill/killall-style broadcast to arbitrary pids.
    owned = {p.pgid for p in popen.procs}
    for pgid, _ in popen.signal_calls:
        assert pgid in owned
    assert all(pgid > 1 for pgid, _ in popen.signal_calls)


# -------------------------------------------------------------------------
# 17. Existing output not overwritten (files genuinely preserved)
# -------------------------------------------------------------------------

def test_existing_summary_not_overwritten(tmp_path):
    out = str(tmp_path / 'bench')
    os.makedirs(out)
    summary = os.path.join(out, 'benchmark_summary.json')
    with open(summary, 'w') as h:
        h.write('{"keep": "this"}')
    code = mb.main(['--output-dir', out])
    # main() refuses before any run; the prior file must be untouched.
    assert code == 2
    with open(summary) as h:
        assert h.read() == '{"keep": "this"}'


def test_existing_nonempty_dir_refuses_execute(tmp_path):
    out = str(tmp_path / 'bench')
    os.makedirs(out)
    os.makedirs(os.path.join(out, 'stale_bag'))
    with open(os.path.join(out, 'stale_bag', 'metadata.yaml'), 'w') as h:
        h.write('prior')
    code = mb.main(['--output-dir', out, '--execute'])
    assert code == 2
    # The stale bag is preserved, not overwritten.
    with open(os.path.join(out, 'stale_bag', 'metadata.yaml')) as h:
        assert h.read() == 'prior'


# --------------------------------------------------------------------------
# 18-19. Aggregate pass/fail; JSON fields
# --------------------------------------------------------------------------

def test_aggregate_overall_pass_fail():
    popen = PopenHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_agg',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        code = runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    assert code == 0
    agg = runner._aggregate()
    assert agg['overall_pass'] is True
    assert agg['runs_passed'] == 2
    assert agg['evaluated_run_count'] == 2


def test_json_aggregate_fields_and_calcs(tmp_path):
    popen = PopenHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    out = str(tmp_path / 'bench')
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir=out,
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    agg = runner._aggregate()
    am = agg['aggregate_metrics']
    assert am['mean_completion_time_s'] == 120.0
    assert am['max_completion_time_s'] == 120.0
    assert abs(am['min_coverage_percent'] - 98.5) < 1e-9
    assert am['total_recovery_requests'] == 0
    assert abs(am['max_filtered_position_error_m'] - 0.05) < 1e-9
    assert abs(am['max_filtered_yaw_error_deg'] - 0.4) < 1e-9
    assert abs(am['max_map_to_odom_translation_step_m'] - 0.10) < 1e-9
    assert abs(am['max_map_to_odom_yaw_step_deg'] - 0.2) < 1e-9
    assert agg['per_run'][0]['recovery_requests'] == 0


# --------------------------------------------------------------------------
# 20. Markdown generation
# --------------------------------------------------------------------------

def test_markdown_report_generation():
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG, output_dir='/tmp/bench_md',
        execute=False, popen=PopenHarness(),
    )
    runner.run_records = [
        {
            'pose_name': 'nominal', 'status': 'passed', 'passed': True,
            'launch_args': ['ros2', 'launch'],
            'recorder_args': ['ros2', 'bag'],
            'completion_time_s': 120.0,
            'evaluator_result': dict(PASS_RESULT),
        },
        {
            'pose_name': 'translated_south', 'status': 'failed',
            'passed': False, 'launch_args': ['ros2', 'launch'],
            'recorder_args': ['ros2', 'bag'],
            'completion_time_s': None,
            'evaluator_result': {
                **PASS_RESULT,
                'passed': False,
                'failure_reasons': ['known map 90.00% < 98.0%'],
            },
        },
    ]
    agg = runner._aggregate()
    md = runner._markdown(agg)
    assert '# V16 Mission Benchmark Summary' in md
    assert 'nominal' in md and 'translated_south' in md
    # Failure reasons rendered from the same aggregate data.
    assert 'known map 90.00% < 98.0%' in md
    assert 'mean_completion_time_s' in md


# --------------------------------------------------------------------------
# 21. Recovery count does not fail a run
# --------------------------------------------------------------------------

def test_recovery_count_does_not_fail_run():
    result = dict(PASS_RESULT)
    result['recovery_requests'] = 5
    result['temporary_failure_events'] = 2
    popen = PopenHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_rec',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (result, True, [])
    try:
        code = runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    assert code == 0
    assert runner.run_records[0]['passed'] is True


# --------------------------------------------------------------------------
# 22-23. enable_rviz defaults; spawn forwarding
# --------------------------------------------------------------------------

def _load_launch(path):
    spec = importlib.util.spec_from_file_location('lt', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declare_args(desc):
    return {
        e.name: e
        for e in desc.entities
        if type(e).__name__ == 'DeclareLaunchArgument'
    }


def _nodes_by_name(desc, package):
    found = []
    for e in desc.entities:
        if type(e).__name__ != 'Node':
            continue
        pkg = getattr(e, '_Node__package', None)
        if pkg == package:
            found.append(e)
    return found


def _node_name(e):
    return getattr(e, '_Node__node_name', None)


def _declare_default(args, name):
    """Perform a DeclareLaunchArgument default_value to a plain string."""
    arg = args[name]
    dv = arg.default_value
    if isinstance(dv, list):
        return ''.join(
            s.perform(None) if hasattr(s, 'perform') else str(s)
            for s in dv
        )
    if hasattr(dv, 'perform'):
        return dv.perform(None)
    return str(dv)


def _includes(desc):
    return [
        e for e in desc.entities
        if type(e).__name__ == 'IncludeLaunchDescription'
    ]


def _include_arg_keys(include):
    return dict(
        include._IncludeLaunchDescription__launch_arguments
    ).keys()


def _eval_condition(condition, enable_rviz_value):
    """Evaluate an IfCondition against enable_rviz = value."""
    from launch import LaunchContext
    ctx = LaunchContext()
    ctx.launch_configurations.update({'enable_rviz': enable_rviz_value})
    return condition.evaluate(ctx)


def test_single_rviz_node_owned_by_display():
    # Exactly one RViz node exists in the whole stack, and it lives in
    # display.launch.py (conditioned on enable_rviz). exploration and
    # sim must not declare their own RViz node.
    display = _load_launch(
        os.path.join(
            SRC_DIR, 'rover_description', 'launch', 'display.launch.py'
        )
    )
    sim = _load_launch(
        os.path.join(SRC_DIR, 'rover_description', 'launch', 'sim.launch.py')
    )
    exp = _load_launch(
        os.path.join(PKG_DIR, 'launch', 'exploration.launch.py')
    )
    display_desc = display.generate_launch_description()
    exp_desc = exp.generate_launch_description()
    sim_desc = sim.generate_launch_description()

    # exploration.launch.py must NOT contain an RViz node.
    assert _nodes_by_name(exp_desc, 'rviz2') == []
    # sim must NOT contain an RViz node either (it forwards only).
    assert _nodes_by_name(sim_desc, 'rviz2') == []
    # display owns exactly one RViz node.
    rviz_nodes = _nodes_by_name(display_desc, 'rviz2')
    assert len(rviz_nodes) == 1
    rviz = rviz_nodes[0]
    assert rviz.condition is not None
    # Default enable_rviz=true makes the RViz node active.
    assert _eval_condition(rviz.condition, 'true') is True
    assert _eval_condition(rviz.condition, 'false') is False
    # Robot-state publisher and bridge remain unconditional.
    assert _nodes_by_name(display_desc, 'robot_state_publisher')[0].condition \
        is None
    assert _nodes_by_name(display_desc, 'ros_gz_bridge')[0].condition is None


def test_enable_rviz_forwarding_chain():
    # enable_rviz is declared in exploration, forwarded to sim, then to
    # display -- the full chain.
    exp = _load_launch(
        os.path.join(PKG_DIR, 'launch', 'exploration.launch.py')
    )
    sim = _load_launch(
        os.path.join(SRC_DIR, 'rover_description', 'launch', 'sim.launch.py')
    )
    display = _load_launch(
        os.path.join(
            SRC_DIR, 'rover_description', 'launch', 'display.launch.py'
        )
    )
    exp_desc = exp.generate_launch_description()
    sim_desc = sim.generate_launch_description()
    display_desc = display.generate_launch_description()

    assert 'enable_rviz' in _declare_args(exp_desc)
    assert 'enable_rviz' in _declare_args(sim_desc)
    assert 'enable_rviz' in _declare_args(display_desc)

    # exploration forwards enable_rviz (and spawn coords) into sim.
    sim_include = [
        i for i in _includes(exp_desc)
        if 'spawn_x' in _include_arg_keys(i)
    ][0]
    assert 'enable_rviz' in _include_arg_keys(sim_include)

    # sim forwards enable_rviz into its display include.
    display_include = [
        i for i in _includes(sim_desc)
        if _include_arg_keys(i) == {'enable_rviz'}
    ][0]
    assert 'enable_rviz' in _include_arg_keys(display_include)


def test_benchmark_disables_rviz():
    cmd = mb.build_launch_command(
        VALID_CONFIG['poses'][0], enable_rviz=False
    )
    assert 'enable_rviz:=false' in cmd


def test_enable_rviz_default_true_interactive():
    # The benchmark command does not override enable_rviz; but the
    # launch default must be true so interactive runs get RViz.
    exp = _load_launch(
        os.path.join(PKG_DIR, 'launch', 'exploration.launch.py')
    )
    assert _declare_default(_declare_args(exp.generate_launch_description()),
                            'enable_rviz') == 'true'
    raw = mb.build_launch_command(
        VALID_CONFIG['poses'][1], enable_rviz=False
    )
    idx_y = raw.index('spawn_y:=-2.0')
    assert raw[idx_y] == 'spawn_y:=-2.0'
    # yaw90 pose
    yaw_pose = {
        'name': 'y', 'spawn_x': 0.0, 'spawn_y': -2.0,
        'spawn_z': 0.02, 'spawn_yaw': 1.5707963267948966,
    }
    cmd = mb.build_launch_command(yaw_pose, enable_rviz=False)
    assert 'spawn_yaw:=1.5707963267948966' in cmd


# --------------------------------------------------------------------------
# 24. Install: YAML + entry point
# --------------------------------------------------------------------------

def test_config_and_entry_point_installed():
    # YAML config exists in package source (installed via glob).
    assert os.path.isfile(CFG)
    # Entry point is declared.
    import rover_exploration
    entry = rover_exploration.__file__.replace(
        '__init__.py', 'mission_benchmark.py'
    )
    assert os.path.isfile(entry)


# -------------------------------------------------------------------------
# 25. Live provider: clock-start / clock-stopped guards
# -------------------------------------------------------------------------

def test_clock_not_started_returns_failure():
    popen = PopenHarness()
    # Provider never emits /clock.
    events = []
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_noclock',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    assert runner.run_records[0]['status'] == 'clock_not_started'


def test_clock_stopped_detected_after_false():
    popen = PopenHarness()
    # Clock advances to 10 then stops; a false is observed first.
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'clock', 'sim': 10.0},
        # no further clock -> stall
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_clockstop',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    assert runner.run_records[0]['status'] == 'clock_stopped'


# -------------------------------------------------------------------------
# 26. Process-group cleanup: real group liveness
# -------------------------------------------------------------------------

def test_leader_exits_children_remain_cleanup_signals_group():
    popen = PopenHarness(kill_group_on_kill=False)
    # Completing run: leader exits on SIGINT but group lingers until
    # SIGKILL; cleanup must escalate and confirm the whole group gone.
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_leader',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
        now=_fake_now(start=0.0, step=0.001),
        clock_stall_wall_s=3.0,
        sleep=lambda *a, **k: None,
    )
    runner._flush_timeout_s = 0.05
    runner._launch_shutdown_timeout_s = 0.05
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (dict(PASS_RESULT), True, [])
    try:
        code = runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    # The run completed (false->true observed) and was evaluated cleanly
    # (the whole process group went away after escalation).
    assert code == 0
    assert runner.run_records[0]['passed'] is True
    assert runner.run_records[0]['status'] in ('completed', 'passed')
    # Each owned group received SIGINT, SIGTERM, and SIGKILL in order.
    rec_signals = [s for (p, s) in popen.signal_calls if p == 100]
    launch_signals = [s for (p, s) in popen.signal_calls if p == 101]
    assert signal.SIGINT in rec_signals
    assert signal.SIGTERM in rec_signals
    assert signal.SIGKILL in rec_signals
    assert launch_signals == rec_signals


def test_cleanup_failure_stops_matrix():
    popen = PopenHarness(kill_group_on_kill=False)
    # Even SIGKILL cannot remove the group (modeled by never setting
    # _group_dead on SIGKILL). Override killpg so SIGKILL is a no-op.

    def noop_killpg(pgid, signum):
        if signum == 0:
            for proc in popen.procs:
                if proc.pgid == pgid:
                    if proc._group_dead:
                        raise ProcessLookupError('gone')
                    return
            raise ProcessLookupError('unknown')
        # SIGINT/SIGTERM/SIGKILL never kill the group in this scenario.
        for proc in popen.procs:
            if proc.pgid == pgid:
                proc._leader_rc = -signum
                return
        raise ProcessLookupError('unknown')

    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_cleanfail',
        execute=True,
        popen=popen,
        killpg=noop_killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
        now=_fake_now(start=0.0, step=0.001),
        clock_stall_wall_s=3.0,
        sleep=lambda *a, **k: None,
    )
    runner._flush_timeout_s = 0.05
    runner._launch_shutdown_timeout_s = 0.05
    code = runner.run()
    # Cleanup failed -> matrix halted after the first pose, not passed.
    assert code == 1
    assert len(runner.run_records) == 1
    assert runner.run_records[0]['status'] == 'cleanup_failed'
    assert runner.run_records[0]['passed'] is False


def test_no_unowned_pgid_signaled():
    popen = PopenHarness()
    events = [{'type': 'clock', 'sim': 0.0}]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_owned2',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
        now=_fake_now(start=0.0, step=1.0),
        clock_stall_wall_s=3.0,
    )
    runner.run()
    owned = {p.pgid for p in popen.procs}
    for pgid, _ in popen.signal_calls:
        assert pgid in owned


# -------------------------------------------------------------------------
# 27. CLI injects the production provider
# -------------------------------------------------------------------------

def test_cli_injects_production_provider(monkeypatch, tmp_path):
    # No Gazebo/ROS launched. Confirm the --execute CLI path requests the
    # production LiveCompletionProvider (proving the runner is wired to a
    # real rclpy observer, not the missing/None provider from V16).
    captured = {}

    class SpyRunner(mb.BenchmarkRunner):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

        def run(self):
            # Capture only; never launch real processes in the test.
            return 0

    monkeypatch.setattr(mb, 'BenchmarkRunner', SpyRunner)
    # Mark the production provider so we can detect it was requested.
    marker = object()
    monkeypatch.setattr(mb, 'LiveCompletionProvider', marker)

    out = str(tmp_path / 'cli')
    code = mb.main(['--output-dir', out, '--execute'])
    # main() must request the production provider on the live path.
    assert captured.get('make_provider') is marker
    # Dry-run must NOT request the provider.
    captured.clear()
    code2 = mb.main(['--output-dir', out + '_dry'])
    assert captured.get('make_provider') is None
    assert code == 0 and code2 == 0


# -------------------------------------------------------------------------
# 28. Report fields: stopping + lifecycle + full result
# -------------------------------------------------------------------------

def test_report_includes_stopping_fields_and_lifecycle():
    popen = PopenHarness()
    result = dict(PASS_RESULT)
    result['active_cmd_vel_after_completion'] = 0
    result['active_cmd_vel_raw_after_completion'] = 0
    result['nonempty_paths_after_completion'] = 0
    result['ground_truth_motion_after_completion_m'] = 0.0
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir='/tmp/bench_fields',
        execute=True,
        popen=popen,
        killpg=popen.killpg,
        getpgid=popen.getpgid,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
    )
    real_read = mb.me.read_bag
    real_eval = mb.me.evaluate_mission
    mb.me.read_bag = lambda b: {}
    mb.me.evaluate_mission = lambda c: (result, True, [])
    try:
        runner.run()
    finally:
        mb.me.read_bag = real_read
        mb.me.evaluate_mission = real_eval
    agg = runner._aggregate()
    per = agg['per_run'][0]
    assert per['evaluator_result'] == result
    assert per['active_cmd_vel_after_completion'] == 0
    assert per['active_cmd_vel_raw_after_completion'] == 0
    assert per['nonempty_paths_after_completion'] == 0
    assert per['ground_truth_motion_after_completion_m'] == 0.0
    assert per['lifecycle_error'] is None
    # Full effective config + spawn values present.
    assert agg['config_used']['poses'][0]['spawn_y'] == 0.0
    assert agg['selected_poses'][0]['name'] == 'nominal'


def test_markdown_reports_lifecycle_error_not_none():
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG, output_dir='/tmp/bench_mdlc',
        execute=False, popen=PopenHarness(),
    )
    runner.run_records = [
        {
            'pose_name': 'nominal', 'status': 'clock_not_started',
            'passed': False, 'launch_args': ['ros2', 'launch'],
            'recorder_args': ['ros2', 'bag'],
            'completion_time_s': None,
            'error': 'simulation clock never started',
            'evaluator_result': None,
        },
    ]
    agg = runner._aggregate()
    md = runner._markdown(agg)
    assert 'clock_not_started' in md
    assert 'simulation clock never started' in md
    # Must not print the placeholder 'None' for a failed-before-eval run.
    assert 'error: none' not in md


# -------------------------------------------------------------------------
# 29. Dependencies: yaml declared; rviz2 owned by rover_description
# -------------------------------------------------------------------------

def test_yaml_dependency_declared():
    pkg_xml = os.path.join(PKG_DIR, 'package.xml')
    with open(pkg_xml) as h:
        content = h.read()
    assert 'python3-yaml' in content
    # rover_exploration no longer owns rviz2 (display.launch.py does).
    assert 'rviz2' not in content


# -------------------------------------------------------------------------
# 30. V16.2: leader reaping + group-liveness correctness
# -------------------------------------------------------------------------

def _run_with_harness(monkeypatch, tmp_path, popen, events, **runner_kwargs):
    monkeypatch.setattr(mb.me, 'read_bag', lambda bag_dir: {})
    monkeypatch.setattr(
        mb.me, 'evaluate_mission', lambda collected: ({'known_map_percent': 100.0}, True, [])
    )
    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir=str(tmp_path / 'bench'),
        execute=True,
        popen=popen,
        getpgid=popen.getpgid,
        killpg=popen.killpg,
        make_provider=lambda: mb.FakeCompletionStream(list(events)),
        now=_fake_now(start=0.0, step=0.001),
        clock_stall_wall_s=3.0,
        sleep=lambda *a, **k: None,
        **runner_kwargs,
    )
    runner._flush_timeout_s = 0.05
    runner._launch_shutdown_timeout_s = 0.05
    code = runner.run()
    return runner, code


def test_reaped_leader_group_gone_without_false_escalation(monkeypatch, tmp_path):
    # Leader is a zombie (group visible pre-reap); once poll() reaps it the
    # group is gone. Cleanup must succeed and must NOT falsely escalate to
    # SIGKILL / report cleanup_failed.
    popen = ReapingHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner, code = _run_with_harness(monkeypatch, tmp_path, popen, events)
    rec = runner.run_records[0]
    assert rec['status'] in ('completed', 'passed')
    assert rec.get('passed') is True
    assert rec.get('cleanup_failed') is not True
    assert rec['status'] != 'cleanup_failed'
    # Group confirmed gone at the SIGINT stage; no SIGTERM/SIGKILL needed.
    rec_signals = [s for (p, s) in popen.signal_calls if p == 100]
    assert rec_signals and rec_signals[0] == signal.SIGINT
    assert signal.SIGKILL not in rec_signals


def test_leader_reaped_descendant_requires_sigkill(monkeypatch, tmp_path):
    # Leader reaped, but a descendant keeps the group alive until SIGKILL.
    popen = DescendantHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner, code = _run_with_harness(monkeypatch, tmp_path, popen, events)
    rec = runner.run_records[0]
    assert rec.get('passed') is True
    # The whole group was signaled, escalating through SIGKILL.
    rec_signals = [s for (p, s) in popen.signal_calls if p == 100 and s != 0]
    assert signal.SIGINT in rec_signals
    assert signal.SIGTERM in rec_signals
    assert signal.SIGKILL in rec_signals
    assert rec_signals[-1] == signal.SIGKILL


def test_sigint_term_fail_sigkill_succeeds_order(monkeypatch, tmp_path):
    popen = DescendantHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner, code = _run_with_harness(monkeypatch, tmp_path, popen, events)
    rec = runner.run_records[0]
    assert rec.get('status') in ('completed', 'passed')
    # Escalation order: SIGINT -> SIGTERM -> SIGKILL for the recorder group.
    rec_signals = [s for (p, s) in popen.signal_calls if p == 100 and s != 0]
    assert rec_signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_cleanup_failure_on_exception_sets_cleanup_failed(monkeypatch, tmp_path):
    # _watch raises; the exception path must run cleanup, which fails, and
    # the run must be recorded as cleanup_failed with the original error kept.
    popen = NeverDiesHarness()

    class BoomProvider:
        def poll(self, timeout=0.2):
            raise RuntimeError('simulated watch failure')

        def shutdown(self):
            pass

    runner = mb.BenchmarkRunner(
        config=VALID_CONFIG,
        output_dir=str(tmp_path / 'bench'),
        execute=True,
        popen=popen,
        getpgid=popen.getpgid,
        killpg=popen.killpg,
        make_provider=lambda: BoomProvider(),
        now=_fake_now(start=0.0, step=0.001),
        clock_stall_wall_s=3.0,
        sleep=lambda *a, **k: None,
    )
    runner._flush_timeout_s = 0.05
    runner._launch_shutdown_timeout_s = 0.05
    runner.run()
    rec = runner.run_records[0]
    assert rec['status'] == 'cleanup_failed'
    assert rec.get('passed') is False
    assert 'cleanup_error' in rec
    assert rec.get('original_error') == 'simulated watch failure'


def test_cleanup_failure_stops_remaining_matrix(monkeypatch, tmp_path):
    # First pose cleanup fails -> only one run attempted, CLI nonzero.
    popen = NeverDiesHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner, code = _run_with_harness(monkeypatch, tmp_path, popen, events)
    assert len(runner.run_records) == 1
    assert runner.run_records[0]['status'] == 'cleanup_failed'
    assert code == 1


def test_no_unowned_pgid_signaled_during_reaping(monkeypatch, tmp_path):
    popen = DescendantHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner, code = _run_with_harness(monkeypatch, tmp_path, popen, events)
    # Only the runner's own process groups are signaled.
    owned_pgids = {proc.pgid for proc in popen.procs}
    for pgid, _ in popen.signal_calls:
        assert pgid in owned_pgids


class _LightLiveProvider(mb.LiveCompletionProvider):
    """LiveCompletionProvider without the rclpy/DDS init (unit-testable)."""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._latest_clock_sim = 0.0
        self._clock_started = False
        self._last_emitted_clock_sim = None
        self._saw_false = False
        self._queue = collections.deque(maxlen=64)

    def shutdown(self):
        pass


def test_queued_completion_returned_before_clock_advance():
    # High-rate /clock must not starve a queued /exploration_complete.
    provider = _LightLiveProvider()
    with provider._lock:
        provider._clock_started = True
        provider._latest_clock_sim = 130.0  # clock advanced far ahead
        provider._queue.append(
            {'type': 'completion', 'sim': 120.0, 'value': True}
        )
    first = provider.poll(0.05)
    assert first is not None and first['type'] == 'completion'
    # The clock event is emitted only on a subsequent poll.
    second = provider.poll(0.05)
    assert second is not None and second['type'] == 'clock'


def test_recorder_signaled_before_launch_during_cleanup(monkeypatch, tmp_path):
    # The recorder group (pid 100) must be signaled before the launch group
    # (pid 101) at every escalation stage.
    popen = DescendantHarness()
    events = [
        {'type': 'clock', 'sim': 0.0},
        {'type': 'completion', 'sim': 1.0, 'value': False},
        {'type': 'completion', 'sim': 120.0, 'value': True},
        {'type': 'clock', 'sim': 130.0},
    ]
    runner, code = _run_with_harness(monkeypatch, tmp_path, popen, events)
    assert runner.run_records[0].get('passed') is True
    # First signal call is the recorder group, not the launch group.
    assert popen.signal_calls and popen.signal_calls[0][0] == 100
    rec_first = popen.signal_calls.index((100, signal.SIGINT))
    launch_first = popen.signal_calls.index((101, signal.SIGINT))
    assert rec_first < launch_first
    rec_kill = popen.signal_calls.index((100, signal.SIGKILL))
    launch_kill = popen.signal_calls.index((101, signal.SIGKILL))
    assert rec_kill < launch_kill


def _dump(obj):
    import tempfile
    import yaml
    path = os.path.join(tempfile.mkdtemp(), 'cfg.yaml')
    with open(path, 'w') as handle:
        yaml.safe_dump(obj, handle)
    return path
