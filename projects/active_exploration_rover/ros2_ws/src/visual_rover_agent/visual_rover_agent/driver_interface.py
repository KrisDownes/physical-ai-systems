"""Model-neutral MCP interface to the bounded ROS rover executor."""

from io import BytesIO
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
Use conservative movements, observe again after movement, and check action results.
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
        self.commands = self.create_publisher(String, '/agent_command', 10)
        self.create_subscription(
            String, '/agent_status', self._status, 10)
        self.create_subscription(
            RosImage, '/camera/image_raw', self._image,
            qos_profile_sensor_data)

    def _image(self, message):
        with self.condition:
            self.frame = message
            self.frame_received = time.monotonic()
            self.condition.notify_all()

    def _status(self, message):
        try:
            status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
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
            while self.frame is None and time.monotonic() < end:
                self.condition.wait(end - time.monotonic())
            if self.frame is None:
                raise ToolError('camera_unavailable')
            if time.monotonic() - self.frame_received > maximum_age:
                raise ToolError('stale_camera')
            return self.frame

    def action(self, action, field=None, value=None, timeout=15.0):
        wall_started = time.monotonic()
        if not self.action_lock.acquire(blocking=False):
            raise ToolError('action_in_progress')
        try:
            command_id = f'rover-{uuid.uuid4().hex}'
            command = {'id': command_id, 'action': action}
            if field:
                command[field] = value
            self.commands.publish(String(data=json.dumps(command)))
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
            self.last_result = result
            return result
        finally:
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


@mcp.tool(description='Return the current rover camera image and observation metadata.')
def observe() -> CallToolResult:
    frame = _topics().observe()
    modes = {'rgb8': 'RGB', 'bgr8': 'BGR', 'rgba8': 'RGBA', 'mono8': 'L'}
    mode = modes.get(frame.encoding)
    if mode is None:
        raise ToolError(f'unsupported_camera_encoding:{frame.encoding}')
    image = Image.frombytes(mode, (frame.width, frame.height), bytes(frame.data))
    if mode == 'BGR':
        red, green, blue = image.split()
        image = Image.merge('RGB', (blue, green, red))
    elif mode == 'RGBA':
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
    return CallToolResult(
        content=[
            ImageContent(type='image', data=__import__('base64').b64encode(
                output.getvalue()).decode(), mimeType='image/jpeg'),
            TextContent(type='text', text=json.dumps(metadata)),
        ],
        structuredContent=metadata,
    )


@mcp.tool(description=(
    'Drive a bounded signed distance using executor safety and odometry. '
    'Positive is forward; negative is reverse.'))
def drive(distance_m: Annotated[
        StrictFloat, Field(ge=-0.5, le=0.5)]) -> dict:
    return _topics().action('drive', 'distance_m', distance_m)


@mcp.tool(description=(
    'Turn a bounded signed angle using executor safety and odometry. '
    'Positive is counterclockwise; negative is clockwise.'))
def turn(angle_deg: Annotated[
        StrictFloat, Field(ge=-90.0, le=90.0)]) -> dict:
    return _topics().action('turn', 'angle_deg', angle_deg)


@mcp.tool(description='Immediately preempt movement through the bounded executor.')
def stop() -> dict:
    return _topics().action('stop')


@mcp.tool(description='Stop safely and record that the driver declares the task finished.')
def finish(summary: str) -> dict:
    result = _topics().action('stop')
    return {'finished': True, 'summary': summary, 'stop_result': result}


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
