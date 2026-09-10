"""Focused tests for the model-neutral MCP boundary."""

import asyncio
import json
from pathlib import Path
from threading import Condition, Lock
import time

from mcp.server.fastmcp.exceptions import ToolError
from sensor_msgs.msg import Image
from std_msgs.msg import String

from visual_rover_agent import driver_interface as driver


class Publisher:
    def __init__(self, owner, reply=True):
        self.owner = owner
        self.reply = reply
        self.messages = []

    def publish(self, message):
        self.messages.append(json.loads(message.data))
        if self.reply:
            command = self.messages[-1]
            with self.owner.condition:
                self.owner.statuses['unrelated'] = {'state': 'succeeded'}
                self.owner.accepted_times[command['id']] = 1.0
                self.owner.statuses[command['id']] = {
                    'id': command['id'], 'state': 'succeeded',
                    'reason': '', 'sim_time_s': 2.0,
                }
                self.owner.condition.notify_all()


class Clock:
    class Now:
        nanoseconds = 1_000_000_000

    def now(self):
        return self.Now()


def bare_topics(reply=True):
    topics = driver.RoverTopics.__new__(driver.RoverTopics)
    topics.condition = Condition()
    topics.action_lock = Lock()
    topics.frame = None
    topics.frame_received = None
    topics.statuses = {}
    topics.accepted_times = {}
    topics.last_result = None
    topics.observation_id = None
    topics.event = lambda *args, **kwargs: None
    topics.commands = Publisher(topics, reply)
    topics.get_clock = lambda: Clock()
    return topics


def test_discovery_and_observe_image_metadata():
    assert {tool.name for tool in driver.mcp._tool_manager.list_tools()} == {
        'observe', 'drive', 'turn', 'stop', 'finish'}
    fake = bare_topics()
    frame = Image()
    frame.width, frame.height, frame.encoding = 2, 1, 'rgb8'
    frame.step = 6
    frame.data = bytes([255, 0, 0, 0, 0, 255])
    frame.header.stamp.sec = 4
    fake.frame = frame
    fake.frame_received = time.monotonic()
    driver.topics = fake
    result = driver.observe()
    assert result.content[0].type == 'image'
    assert result.content[0].mimeType == 'image/jpeg'
    assert result.structuredContent['width'] == 2
    assert result.structuredContent['captured_sim_time_s'] == 4.0


def test_missing_and_stale_camera_are_errors():
    fake = bare_topics()
    try:
        fake.observe(timeout=0.001)
        assert False
    except ToolError as error:
        assert str(error) == 'camera_unavailable'
    fake.frame = Image()
    fake.frame_received = time.monotonic() - 2.0
    try:
        fake.observe(timeout=0.001)
        assert False
    except ToolError as error:
        assert str(error) == 'stale_camera'


def test_action_correlation_timeout_concurrency_and_disconnect():
    fake = bare_topics()
    result = fake.action('drive', 'distance_m', 0.25)
    assert fake.commands.messages[0]['action'] == 'drive'
    assert result['terminal_state'] == 'succeeded'
    assert result['accepted_sim_time_s'] == 1.0
    assert result['terminal_sim_time_s'] == 2.0
    assert result['execution_sim_time_s'] == 1.0
    assert result['tool_wall_time_s'] >= 0.0
    assert 'unrelated' in fake.statuses

    fake = bare_topics(reply=False)
    try:
        fake.action('drive', 'distance_m', 0.1, timeout=0.001)
        assert False
    except ToolError as error:
        assert str(error) == 'action_timeout'
    assert fake.commands.messages[-1]['action'] == 'stop'

    fake.action_lock.acquire()
    try:
        fake.action('turn', 'angle_deg', 10)
        assert False
    except ToolError as error:
        assert str(error) == 'action_in_progress'
    fake.action_lock.release()
    fake.disconnect()
    assert fake.commands.messages[-1]['action'] == 'stop'


def test_interface_never_names_raw_velocity_topic():
    source = open(driver.__file__, encoding='utf-8').read()
    assert '/cmd_vel' not in source


def test_stop_bypasses_motion_lock():
    fake = bare_topics()
    fake.action_lock.acquire()
    result = fake.action('stop')
    assert result['terminal_state'] == 'succeeded'
    assert fake.action_lock.locked()
    fake.action_lock.release()


def test_observation_must_follow_terminal_timestamp():
    fake = bare_topics()
    fake.frame = Image()
    fake.frame.header.stamp.sec = 2
    fake.frame_received = time.monotonic()
    fake.last_result = {'terminal_sim_time_s': 2.0}
    try:
        fake.observe(timeout=0.001)
        assert False
    except ToolError as error:
        assert str(error) == 'post_action_camera_unavailable'
    fake.frame.header.stamp.nanosec = 1
    assert fake.observe() is fake.frame


def test_discovered_motion_schemas_are_bounded_and_strict():
    tools = {tool.name: tool for tool in driver.mcp._tool_manager.list_tools()}
    distance = tools['drive'].parameters['properties']['distance_m']
    angle = tools['turn'].parameters['properties']['angle_deg']
    assert (distance['minimum'], distance['maximum']) == (-0.5, 0.5)
    assert (angle['minimum'], angle['maximum']) == (-90.0, 90.0)

    fake = bare_topics()
    driver.topics = fake
    for value in (True, float('nan'), float('inf')):
        try:
            asyncio.run(driver.mcp._tool_manager.call_tool(
                'drive', {'distance_m': value}))
            assert False
        except ToolError:
            pass
    assert fake.commands.messages == []


def test_versioned_client_configuration_has_no_personal_checkout_path():
    workspace = Path(__file__).parents[3]
    files = [workspace / '.codex/config.toml', *(
        workspace / 'config').glob('*')]
    assert all('/home/krisd' not in path.read_text() for path in files)


def test_mcp_stop_interrupts_pending_drive():
    """Exercise concurrent registered MCP handlers, not just the transport lock."""
    fake = bare_topics(reply=False)
    original_publish = fake.commands.publish

    def publish(message):
        original_publish(message)
        command = json.loads(message.data)
        if command['action'] == 'stop':
            with fake.condition:
                for sent in fake.commands.messages:
                    fake.accepted_times[sent['id']] = 1.
                    fake.statuses[sent['id']] = {
                        'state': 'succeeded' if sent['action'] == 'stop' else 'aborted',
                        'reason': 'stopped', 'sim_time_s': 2.}
                fake.condition.notify_all()

    fake.commands.publish = publish
    driver.topics = fake

    async def exercise():
        movement = asyncio.create_task(driver.mcp._tool_manager.call_tool(
            'drive', {'distance_m': .3}))
        for _ in range(100):
            if fake.commands.messages:
                break
            await asyncio.sleep(.01)
        assert fake.commands.messages
        await asyncio.wait_for(driver.mcp._tool_manager.call_tool('stop', {}), 1.)
        await asyncio.wait_for(movement, 1.)

    asyncio.run(exercise())
    assert [m['action'] for m in fake.commands.messages] == ['drive', 'stop']


def test_camera_padding_and_invalid_layout():
    fake = bare_topics()
    frame = Image()
    frame.width, frame.height, frame.encoding, frame.step = 1, 1, 'bgr8', 4
    frame.data = bytes([0, 0, 255, 0])
    fake.frame, fake.frame_received = frame, time.monotonic()
    driver.topics = fake
    assert driver.observe().content[0].type == 'image'
    frame.step = 2
    try:
        driver.observe()
        assert False
    except ToolError as error:
        assert str(error) == 'invalid_camera_layout'


def test_pilot_movement_cap_rejects_before_ros_publish(monkeypatch):
    fake = bare_topics()
    fake.movement_count = 100
    monkeypatch.setenv('ROVER_MOVEMENT_LIMIT', '100')
    try:
        fake.action('drive', 'distance_m', .1)
        assert False
    except ToolError as error:
        assert str(error) == 'movement_limit_reached'
    assert not fake.commands.messages
