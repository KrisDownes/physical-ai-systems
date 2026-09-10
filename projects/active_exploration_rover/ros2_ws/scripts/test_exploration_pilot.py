"""Focused checks of the new observation and pilot boundary, no model calls."""
import asyncio
import json
import math
from types import SimpleNamespace
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from mcp.types import CallToolResult, TextContent
from mcp.server.fastmcp.exceptions import ToolError
from visual_rover_agent.onboard import map_view, clearances
from visual_rover_agent import driver_interface as driver
from exploration_pilot import classify


def test_map_uses_all_unranked_frontiers_and_rotated_origin():
    grid=OccupancyGrid();grid.info.width=5;grid.info.height=5;grid.info.resolution=1.
    grid.info.origin.position.x=10.;grid.info.origin.orientation.z=math.sqrt(.5);grid.info.origin.orientation.w=math.sqrt(.5)
    grid.data=[-1]*5+[0]*15+[-1]*5
    candidates,image=map_view(grid,{'state':'missing'})
    assert len(candidates)==2 and image
    assert candidates[0]['x_m']==8.5 and candidates[0]['y_m']==2.5
    assert candidates[1]['x_m']==6.5


def test_scan_invalid_and_no_return_are_distinct():
    scan=LaserScan();scan.angle_min=-math.pi;scan.angle_increment=math.pi/4
    scan.range_min=.1;scan.range_max=10.;scan.ranges=[float('inf')]*9
    assert clearances(scan)['front']['state']=='no_return_within_range'
    scan.ranges=[float('nan')]*9
    assert clearances(scan)['front']['state']=='missing_or_invalid'


def test_movement_bundles_terminal_and_fresh_observation(monkeypatch):
    monkeypatch.setenv('ROVER_OBSERVATION_MODE','onboard')
    result={'command_id':'a','terminal_sim_time_s':2.,'terminal_state':'succeeded'}
    monkeypatch.setattr(driver,'topics',SimpleNamespace(action=lambda *a:result))
    monkeypatch.setattr(driver,'observe',lambda:CallToolResult(content=[TextContent(type='text',text='fresh')]))
    value=asyncio.run(driver.movement('turn','angle_deg',5.))
    assert json.loads(value.content[0].text)==result
    assert value.content[1].text=='fresh'
    def unavailable():raise ToolError('post_action_camera_unavailable')
    monkeypatch.setattr(driver,'observe',unavailable)
    value=asyncio.run(driver.movement('turn','angle_deg',5.))
    assert value['command_id']=='a' and value['observation_error']=='post_action_camera_unavailable'


def test_quota_requires_service_error_not_signal_or_model_claim():
    assert classify('interrupted','signal 15',False,False)=='interrupted'
    assert classify('driver_exit',json.dumps({'type':'item.completed','item':{'text':'usage limit'}}),False,False)=='navigation_failure'
    assert classify('driver_exit',json.dumps({'type':'error','message':'usage limit reached'}),False,False)=='quota_exhaustion'


def test_discovery_must_resolve_names_and_reject_extra_publishers():
    from exploration_pilot import controller_ready
    graph=dict(velocity_publishers=['obstacle_guard','experiment_supervisor'],
               command_publishers=['experiment_supervisor'],raw_velocity_publishers=['path_follower'])
    assert controller_ready(graph,'classical')
    for key in graph:
        unknown={**graph,key:['_NODE_NAME_UNKNOWN_']}
        assert not controller_ready(unknown,'classical')
        assert not controller_ready({**graph,key:graph[key]+['other']},'classical')
    llm=dict(velocity_publishers=['agent_executor','experiment_supervisor'],
             command_publishers=['experiment_supervisor','rover_driver_interface'],raw_velocity_publishers=[])
    assert controller_ready(llm,'llm')
    assert not controller_ready({**llm,'command_publishers':llm['command_publishers']+['rover_driver_interface']},'llm')


def test_classical_method_selection_cannot_launch_llm():
    from exploration_pilot import selected_methods
    assert selected_methods('classical')==('classical',)
    import pytest
    with pytest.raises(ValueError):selected_methods('classical',True)


def test_luna_revision_preserves_base_prompt_and_single_method():
    import hashlib
    from exploration_pilot import TASK, selected_methods
    base, addition = TASK.rsplit('\n', 1)
    assert hashlib.sha256(base.encode()).hexdigest() == '02d4a493fb15c563117b8fe6d53a918da213bf643558f101410aeaf81a7c20b1'
    assert hashlib.sha256(TASK.encode()).hexdigest() == 'd082be1f5d8739715bba1f148cbd652523616b02d6498d387404e4838d0aba77'
    assert addition.endswith('Emergency stopping remains available at all times.')
    assert selected_methods('llm') == ('llm',)
