"""Model-neutral MCP interface to the bounded ROS rover executor."""

from io import BytesIO
import asyncio
import json
import os
import sys
from threading import Condition, Lock, Thread
import time
import uuid
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent
from PIL import Image
from pydantic import Field, StrictFloat
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String


INSTRUCTIONS = """Observe before moving and base decisions on the returned image.
Use conservative movements and check action results. Movement responses include fresh observations in onboard mode.
Stop when uncertain. Call finish when the task is done."""
TERMINAL = {'succeeded', 'rejected', 'aborted'}


class RoverTopics(Node):
    """Thread-safe ROS transport with no planning or motion logic."""

    def __init__(self):
        super().__init__('rover_driver_interface')
        self.condition = Condition()
        self.action_lock = Lock()
        self.frame = None
        self.frame_received = None
        self.statuses = {}
        self.accepted_times = {}
        self.last_result = None
        self.observation_id = None
        self.history = []
        self.movement_count = 0
        self.onboard = None
        if os.environ.get("ROVER_OBSERVATION_MODE") == "onboard":
            from visual_rover_agent.onboard import Onboard
            self.onboard = Onboard(self)
        self.timing = self.create_publisher(String, "/agent_timing", 100)
        self.commands = self.create_publisher(String, '/agent_command', 10)
        self.create_subscription(
            String, '/agent_status', self._status, 10)
        self.create_subscription(
            RosImage, '/camera/image_raw', self._image,
            qos_profile_sensor_data)

    def event(self, kind, **data):
        self.timing.publish(String(data=json.dumps(dict(kind=kind, wall_s=time.monotonic(), **data))))

    def _image(self, message):
        with self.condition:
            self.frame = message
            self.frame_received = time.monotonic()
            self.event("camera_received", sim_s=message.header.stamp.sec + message.header.stamp.nanosec / 1e9)
            self.condition.notify_all()

    def _status(self, message):
        try:
            status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(status, dict):
            return
        with self.condition:
            command_id = status.get('id')
            if status.get('state') == 'accepted':
                self.accepted_times[command_id] = status.get('sim_time_s')
            elif status.get('state') in TERMINAL:
                self.statuses[command_id] = status
            else:
                return
            self.condition.notify_all()

    def observe(self, timeout=2.0, maximum_age=1.0):
        with self.condition:
            end = time.monotonic() + timeout
            def ordered():
                if self.frame is None:
                    return False
                stamp = self.frame.header.stamp
                completed = (self.last_result or {}).get('terminal_sim_time_s')
                return completed is None or stamp.sec + stamp.nanosec / 1e9 > completed
            while not ordered() and time.monotonic() < end:
                self.condition.wait(end - time.monotonic())
            if self.frame is None:
                raise ToolError('camera_unavailable')
            if time.monotonic() - self.frame_received > maximum_age:
                raise ToolError('stale_camera')
            if not ordered():
                raise ToolError('post_action_camera_unavailable')
            return self.frame

    def action(self, action, field=None, value=None, timeout=30.0):
        wall_started = time.monotonic()
        command_id = f"rover-{uuid.uuid4().hex}"
        self.event("request", id=command_id, observation_id=self.observation_id, action=action)
        locked = action != 'stop'
        if locked and not self.action_lock.acquire(blocking=False):
            raise ToolError('action_in_progress')
        try:
            if locked:
                limit = int(os.environ.get('ROVER_MOVEMENT_LIMIT', '0'))
                count = getattr(self, 'movement_count', 0)
                if limit and count >= limit:
                    raise ToolError('movement_limit_reached')
                self.movement_count = count + 1
            minimum_subscribers = int(os.environ.get('ROVER_MIN_COMMAND_SUBSCRIBERS', '0'))
            if locked and minimum_subscribers:
                ready_by = time.monotonic() + 3.0
                while self.commands.get_subscription_count() < minimum_subscribers:
                    if time.monotonic() >= ready_by:
                        raise ToolError('command_subscribers_unavailable')
                    time.sleep(0.02)
            command = {'id': command_id, 'action': action}
            if field:
                command[field] = value
            self.commands.publish(String(data=json.dumps(command)))
            self.event("command_published", id=command_id)
            with self.condition:
                end = time.monotonic() + timeout
                while command_id not in self.statuses:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        self._safe_stop()
                        raise ToolError('action_timeout')
                    self.condition.wait(remaining)
                status = self.statuses.pop(command_id)
                accepted_at = self.accepted_times.pop(command_id, None)
            finished_at = status.get('sim_time_s')
            elapsed = 0.0
            if accepted_at is not None and finished_at is not None:
                elapsed = max(0.0, finished_at - accepted_at)
            result = {
                'command_id': command_id,
                'accepted_sim_time_s': accepted_at,
                'terminal_sim_time_s': finished_at,
                'terminal_state': status['state'],
                'reason': status.get('reason', ''),
                'execution_sim_time_s': elapsed,
                'tool_wall_time_s': time.monotonic() - wall_started,
            }
            if field:
                result[field] = value
            with self.condition:
                previous = (self.last_result or {}).get('terminal_sim_time_s')
                if previous is None or (finished_at is not None and finished_at >= previous):
                    self.last_result = result
            if locked and hasattr(self, 'history'):
                self.history.append(dict(action=action, requested=value, state=result['terminal_state'],
                                         reason=result['reason'], sim_s=finished_at))
            return result
        finally:
            if locked:
                self.action_lock.release()

    def _safe_stop(self):
        command = {
            'id': f'rover-stop-{uuid.uuid4().hex}',
            'action': 'stop',
        }
        self.commands.publish(String(data=json.dumps(command)))

    def disconnect(self):
        self._safe_stop()


topics = None
mcp = FastMCP('Open Rover Driver', instructions=INSTRUCTIONS, log_level='ERROR')


def _topics():
    if topics is None:
        raise ToolError('rover_environment_unavailable')
    return topics


@mcp.tool(description='Return a fresh camera observation; configured onboard mode also includes lidar, online SLAM map, estimated pose and unranked frontier candidates.')
def observe() -> CallToolResult:
    started = time.monotonic()
    frame = _topics().observe()
    received = _topics().frame_received
    preparation = time.monotonic()
    formats = {'rgb8': ('RGB', 'RGB', 3), 'bgr8': ('RGB', 'BGR', 3),
               'rgba8': ('RGBA', 'RGBA', 4), 'mono8': ('L', 'L', 1)}
    format_spec = formats.get(frame.encoding)
    if format_spec is None:
        raise ToolError(f'unsupported_camera_encoding:{frame.encoding}')
    mode, raw_mode, channels = format_spec
    if (frame.width <= 0 or frame.height <= 0
            or frame.step < frame.width * channels
            or len(frame.data) != frame.step * frame.height):
        raise ToolError('invalid_camera_layout')
    image = Image.frombytes(mode, (frame.width, frame.height), bytes(frame.data),
                            'raw', raw_mode, frame.step)
    if mode == 'RGBA':
        image = image.convert('RGB')
    output = BytesIO()
    image.save(output, format='JPEG', quality=85)
    stamp = frame.header.stamp
    metadata = {
        'observation_id': f'obs-{uuid.uuid4().hex}',
        'captured_sim_time_s': stamp.sec + stamp.nanosec / 1e9,
        'width': frame.width,
        'height': frame.height,
        'encoding': frame.encoding,
        'last_completed_action_result': _topics().last_result,
    }
    content = [ImageContent(type='image', data=__import__('base64').b64encode(
        output.getvalue()).decode(), mimeType='image/jpeg')]
    if getattr(_topics(), 'onboard', None):
        onboard, map_image = _topics().onboard.snapshot(metadata['captured_sim_time_s'])
        metadata.update(onboard)
        # At most 100 movements in the pilot; retain every compact action record.
        metadata['navigation_history'] = _topics().history
        metadata['image_order'] = ['camera'] + (['online_slam_map'] if map_image else [])
        if map_image:
            content.append(ImageContent(type='image',data=map_image,mimeType='image/png'))
    metadata['camera_state'] = 'fresh'
    metadata['ready_monotonic_s'] = time.monotonic()
    _topics().observation_id = metadata["observation_id"]
    _topics().event("observation_ready", observation_id=metadata["observation_id"], received_wall_s=received, preparation_s=time.monotonic()-preparation, wait_s=preparation-started, sim_s=metadata["captured_sim_time_s"])
    content.append(TextContent(type='text',text=json.dumps(metadata,allow_nan=False)))
    result = CallToolResult(content=content, structuredContent=metadata)
    log = os.environ.get('ROVER_OBSERVATION_LOG')
    if log:
        with open(log,'a') as stream:
            stream.write(result.model_dump_json()+'\n')
    return result


async def movement(action, field, value):
    result = await asyncio.to_thread(_topics().action, action, field, value)
    if os.environ.get('ROVER_OBSERVATION_MODE') != 'onboard':
        return result
    try:
        observation = await asyncio.to_thread(observe)
        # Keep the terminal result even if a sensor recovery is necessary.
        observation.content.insert(0, TextContent(type='text',text=json.dumps(result)))
        return observation
    except ToolError as error:
        return {**result, 'observation_error': str(error), 'recovery': 'call observe'}



@mcp.tool(description=(
    'Drive a bounded signed distance using executor safety and odometry. '
    'Positive is forward; negative is reverse.'))
async def drive(distance_m: Annotated[
        StrictFloat, Field(ge=-0.5, le=0.5)]):
    return await movement('drive', 'distance_m', distance_m)


@mcp.tool(description=(
    'Turn a bounded signed angle using executor safety and odometry. '
    'Positive is counterclockwise; negative is clockwise.'))
async def turn(angle_deg: Annotated[
        StrictFloat, Field(ge=-90.0, le=90.0)]):
    return await movement('turn', 'angle_deg', angle_deg)


@mcp.tool(description='Immediately preempt movement through the bounded executor.')
async def stop() -> dict:
    return await asyncio.to_thread(_topics().action, 'stop')


@mcp.tool(description='Stop safely and record that the driver declares the task finished.')
async def finish(summary: str) -> dict:
    result = await asyncio.to_thread(_topics().action, 'stop')
    return {'finished': result['terminal_state'] == 'succeeded',
            'summary': summary, 'stop_result': result}


def main():
    """Spin ROS in the background while MCP exclusively owns stdout."""
    global topics
    protocol_stdout = os.dup(1)
    os.dup2(2, 1)
    try:
        rclpy.init()
        topics = RoverTopics()
    finally:
        os.dup2(protocol_stdout, 1)
        os.close(protocol_stdout)
    executor = SingleThreadedExecutor()
    executor.add_node(topics)
    thread = Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        minimum_subscribers = int(os.environ.get('ROVER_MIN_COMMAND_SUBSCRIBERS', '0'))
        if minimum_subscribers:
            topics.observe(timeout=20.0)
            ready_by = time.monotonic() + 20.0
            while (topics.commands.get_subscription_count() < minimum_subscribers
                   or topics.count_publishers('/agent_status') < 1):
                if time.monotonic() >= ready_by:
                    raise ToolError('experiment_transport_not_ready')
                time.sleep(.05)
        mcp.run(transport='stdio')
    finally:
        try:
            topics.disconnect()
            time.sleep(0.05)
        except Exception as error:
            print(f'safe-stop failed: {error}', file=sys.stderr)
        executor.shutdown()
        topics.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
