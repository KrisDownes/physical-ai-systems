"""Onboard route-cost annotations only; never selects or modifies a motion plan."""
import math
from rover_exploration.frontier_selection import world_point_to_grid_cell

INPUT_SPEC_VERSION = 'hierarchical-onboard-v3'
INPUT_SCHEMA = {
    'type': 'object',
    'required': ['candidates', 'planned_path_context', 'input_spec_version'],
    'properties': {
        'input_spec_version': {'const': INPUT_SPEC_VERSION},
        'candidates': {'type': 'array', 'items': {
            'type': 'object', 'required': ['id', 'x_m', 'y_m', 'cells', 'planned_path_length_m', 'planned_path_length_state'],
            'properties': {'id': {'type': 'string'}, 'x_m': {'type': 'number'}, 'y_m': {'type': 'number'},
                'cells': {'type': 'integer'}, 'planned_path_length_m': {'type': ['number', 'null'], 'minimum': 0},
                'planned_path_length_state': {'enum': ['available', 'unavailable']},
                'planned_path_length_unavailable_reason': {'type': 'string'}},
        }},
        'planned_path_context': {'type': 'object',
            'required': ['map_captured_sim_time_s', 'pose_captured_sim_time_s', 'frame', 'method'],
            'properties': {'map_captured_sim_time_s': {'type': ['number', 'null']},
                'pose_captured_sim_time_s': {'type': ['number', 'null']},
                'frame': {'type': ['string', 'null']}, 'method': {'const': 'conditioned_grid_bfs_steps_times_resolution'}}},
    },
}


def annotate_route_distances(snapshot, mapping, build_grid, reachable_tree):
    """One tree from the exact visible pose, shared by all unchanged eligible IDs.

    Callbacks are the existing controller's _build_planning_grid/_reachable_tree.
    Its selection tree may use an older map-stamped TF pose, so do not reuse that
    tree without exact start/map parity. This tree only annotates the snapshot.
    """
    pose = snapshot.get('estimated_pose', {})
    map_state = snapshot.get('slam_map', {})
    result = dict(snapshot, candidates=[dict(c) for c in snapshot['candidates']], input_spec_version=INPUT_SPEC_VERSION)
    actual_stamp = (mapping.header.stamp.sec + mapping.header.stamp.nanosec / 1e9) if mapping else None
    pose_stamp = pose.get('captured_sim_time_s')
    result['planned_path_context'] = dict(map_captured_sim_time_s=actual_stamp,
        pose_captured_sim_time_s=pose_stamp if isinstance(pose_stamp, (int,float)) and math.isfinite(pose_stamp) else None,
        frame=mapping.header.frame_id if mapping else None, method='conditioned_grid_bfs_steps_times_resolution')
    tree = None; reason = None
    if mapping is None or map_state.get('state') != 'valid':
        reason = 'map_missing_or_invalid'
    elif pose.get('state') != 'valid' or any(not isinstance(pose.get(k),(int,float)) or not math.isfinite(pose[k]) for k in ('x_m','y_m','captured_sim_time_s')):
        reason = 'pose_missing_or_invalid'
    elif actual_stamp != map_state.get('captured_sim_time_s'):
        reason = 'map_timestamp_mismatch'
    elif mapping.header.frame_id != 'map' or pose.get('frame') != 'map' or any(abs(getattr(mapping.info.origin.orientation,k))>1e-6 for k in ('x','y','z')):
        reason = 'unsupported_frame'
    elif not math.isfinite(mapping.info.resolution) or mapping.info.resolution <= 0:
        reason = 'invalid_resolution'
    else:
        info = mapping.info
        start = world_point_to_grid_cell(pose['x_m'],pose['y_m'],info.resolution,info.origin.position.x,info.origin.position.y)
        grid, _ = build_grid(mapping)
        tree = reachable_tree(mapping.data,grid,info.width,info.height,start)
        if tree is None: reason = 'start_unreachable'
    for c in result['candidates']:
        cost = None; unavailable = reason
        if tree is not None:
            if all(isinstance(c.get(k),(int,float)) and math.isfinite(c[k]) for k in ('x_m','y_m')):
                cell = world_point_to_grid_cell(c['x_m'],c['y_m'],info.resolution,info.origin.position.x,info.origin.position.y)
                steps = tree['cost'].get(cell)
                if isinstance(steps,(int,float)) and math.isfinite(steps) and steps >= 0:
                    cost = steps * info.resolution
                if cost is None or not math.isfinite(cost): cost = None; unavailable = 'missing_or_nonfinite_route_cost'
            else: unavailable = 'invalid_candidate_position'
        c['planned_path_length_m'] = cost
        c['planned_path_length_state'] = 'available' if cost is not None else 'unavailable'
        if cost is None: c['planned_path_length_unavailable_reason'] = unavailable
    return result
