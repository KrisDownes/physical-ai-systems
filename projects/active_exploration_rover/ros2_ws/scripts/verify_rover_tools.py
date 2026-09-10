"""Fixed transport diagnostics, explicitly not a navigation policy or episode."""
import asyncio
import json
import os
from pathlib import Path
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main():
    parameters = StdioServerParameters(command=str(ROOT/'scripts/rover_driver_mcp'), env=dict(os.environ))
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {'observe','drive','turn','stop','finish'}
            index = 0

            async def call(name, arguments=None):
                nonlocal index
                identifier = str(index)
                index += 1
                arguments = arguments or {}
                result = await session.call_tool(name, arguments)
                print(json.dumps({'type':'item.completed', 'item':{
                    'id':identifier, 'type':'mcp_tool_call', 'server':'rover',
                    'tool':name, 'arguments':arguments, 'result':result.model_dump(mode='json'),
                    'error':None, 'status':'completed'}}), flush=True)
                assert not result.isError, result
                for content in result.content:
                    if content.type == 'text':
                        return json.loads(content.text)

            try:
                await call('observe')
                start = time.monotonic()
                pending = asyncio.create_task(call('drive', {'distance_m': .4}))
                await asyncio.sleep(.3)
                stopped = await call('stop')
                movement = await asyncio.wait_for(pending, 3)
                assert movement['terminal_state'] == 'aborted', movement
                assert movement['reason'] == 'stopped', movement
                assert stopped['terminal_state'] == 'succeeded', stopped
                assert time.monotonic()-start < 3
                observation = await call('observe')
                assert observation['captured_sim_time_s'] > stopped['terminal_sim_time_s']
                turn = await call('turn', {'angle_deg': 10.})
                assert turn['terminal_state'] == 'succeeded', turn
                observation = await call('observe')
                assert observation['captured_sim_time_s'] > turn['terminal_sim_time_s']
                print(json.dumps({'type':'item.completed', 'item':{'type':'agent_message',
                      'text':'PASS: live MCP stop interrupted drive; bounded turn completed; post-action frames ordered.'}}), flush=True)
            finally:
                await call('finish', {'summary':'Fixed tool-boundary diagnostic completed; not a model navigation run.'})


if __name__ == '__main__':
    asyncio.run(main())
