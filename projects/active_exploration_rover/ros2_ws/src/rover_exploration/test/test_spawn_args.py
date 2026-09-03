"""
Structural tests for configurable robot spawn arguments.

Parses the launch files directly (no ROS runtime) and verifies the spawn
pose arguments exist, default to the V14 values, and flow from
exploration.launch.py into sim.launch.py with the correct Gazebo create
flags (-x, -y, -z, -Y).
"""

import importlib.util
import os

from launch import LaunchContext
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROVER_EXPLORATION = os.path.dirname(HERE)
SRC_DIR = os.path.dirname(ROVER_EXPLORATION)


def _load(path):
    spec = importlib.util.spec_from_file_location('launch_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _declared_args(desc):
    names = []
    defaults = {}
    for entity in desc.entities:
        if type(entity).__name__ == 'DeclareLaunchArgument':
            names.append(entity.name)
            dv = entity.default_value
            if isinstance(dv, list):
                defaults[entity.name] = ' '.join(
                    s.perform(None) if hasattr(s, 'perform') else str(s)
                    for s in dv
                )
            elif hasattr(dv, 'perform'):
                defaults[entity.name] = dv.perform(None)
            else:
                defaults[entity.name] = str(dv)
    return names, defaults


def test_sim_launch_declares_spawn_defaults():
    sim = _load(
        os.path.join(SRC_DIR, 'rover_description', 'launch', 'sim.launch.py')
    )
    names, defaults = _declared_args(sim.generate_launch_description())
    for arg in ('spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw'):
        assert arg in names
    assert str(defaults['spawn_x']) == '0.0'
    assert str(defaults['spawn_y']) == '0.0'
    assert str(defaults['spawn_z']) == '0.02'
    assert str(defaults['spawn_yaw']) == '0.0'


def test_sim_launch_passes_flags_to_create():
    sim = _load(
        os.path.join(SRC_DIR, 'rover_description', 'launch', 'sim.launch.py')
    )
    desc = sim.generate_launch_description()
    for entity in desc.entities:
        node = getattr(entity, 'action', entity)
        if type(node).__name__ == 'Node' and getattr(
            node, '_Node__package', None
        ) == 'ros_gz_sim':
            arguments = [str(a) for a in node._Node__arguments]
            # Flags must appear.
            assert '-x' in arguments
            assert '-y' in arguments
            assert '-z' in arguments
            assert '-Y' in arguments
            # Each spawn arg is wired through its own LaunchConfiguration.
            configs = [
                a for a in node._Node__arguments
                if type(a).__name__ == 'LaunchConfiguration'
            ]
            assert len(configs) == 4
            config_names = set()
            for c in configs:
                name_sub = c.variable_name[0]
                config_names.add(name_sub.perform(None))
            assert config_names == {
                'spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw',
            }
            return
    pytest.fail('ros_gz_sim create node not found')


def test_exploration_launch_declares_and_forwards_spawn():
    exp = _load(
        os.path.join(
            SRC_DIR, 'rover_exploration', 'launch', 'exploration.launch.py'
        )
    )
    desc = exp.generate_launch_description()
    names, defaults = _declared_args(desc)
    for arg in ('spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw'):
        assert arg in names
        assert str(defaults[arg]) in ('0.0', '0.02')

    # The simulation include must forward all four spawn args.
    for entity in desc.entities:
        node = getattr(entity, 'action', entity)
        if type(node).__name__ == 'IncludeLaunchDescription':
            forwarded = (
                node._IncludeLaunchDescription__launch_arguments
            )
            keys = dict(forwarded).keys()
            for arg in ('spawn_x', 'spawn_y', 'spawn_z', 'spawn_yaw'):
                assert arg in keys
            return
    pytest.fail('simulation include not found')


def test_agent_mode_excludes_classical_policy_and_motion_nodes():
    """Agent mode must own motion without running classical policy nodes."""
    exp = _load(os.path.join(
        SRC_DIR, 'rover_exploration', 'launch', 'exploration.launch.py'))
    context = LaunchContext()
    context.launch_configurations['agent_mode'] = 'true'
    context.launch_configurations['enable_motion'] = 'true'
    found = {}
    for entity in exp.generate_launch_description().entities:
        if type(entity).__name__ != 'Node':
            continue
        package = entity._Node__package
        executable = entity.node_executable
        package = package.perform(context) if hasattr(package, 'perform') else package
        executable = (
            executable.perform(context)
            if hasattr(executable, 'perform') else executable
        )
        enabled = (
            True if entity.condition is None
            else entity.condition.evaluate(context)
        )
        found[(str(package), str(executable))] = enabled

    assert found[('rover_exploration', 'frontier_detector')] is False
    assert found[('rover_control', 'path_follower')] is False
    assert found[('rover_control', 'obstacle_guard')] is False
    assert found[('visual_rover_agent', 'agent_executor')] is True
