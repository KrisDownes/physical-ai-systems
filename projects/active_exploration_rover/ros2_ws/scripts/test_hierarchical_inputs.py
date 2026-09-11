"""Focused route annotation checks; no ROS node or model execution."""
import copy, math
from types import SimpleNamespace as NS
from hierarchical_inputs import annotate_route_distances


def fixture():
    m=NS(header=NS(stamp=NS(sec=2,nanosec=0),frame_id='map'), info=NS(resolution=.5,width=3,height=2,
        origin=NS(position=NS(x=0.,y=0.),orientation=NS(x=0.,y=0.,z=0.))),data=[0]*6)
    s=dict(candidates=[dict(id='A2',x_m=.75,y_m=.25,cells=5),dict(id='A1',x_m=1.25,y_m=.25,cells=7)],
        estimated_pose=dict(state='valid',frame='map',x_m=.25,y_m=.25,captured_sim_time_s=2.1),
        slam_map=dict(state='valid',captured_sim_time_s=2.))
    return m,s


def test_route_cost_is_tree_distance_and_does_not_change_candidates():
    m,s=fixture();before=copy.deepcopy(s);calls=[]
    def tree(*args):calls.append(args);return {'cost':{(0,1):3,(0,2):4}}
    out=annotate_route_distances(s,m,lambda _:([0]*6,0),tree)
    assert len(calls)==1 and s==before
    assert [c['id'] for c in out['candidates']]==['A2','A1']
    assert [c['planned_path_length_m'] for c in out['candidates']]==[1.5,2.]
    assert out['planned_path_context']['pose_captured_sim_time_s']==2.1
    assert out['planned_path_context']['map_captured_sim_time_s']==2.


def test_missing_nonfinite_and_invalid_inputs_are_explicitly_unavailable():
    m,s=fixture()
    for tree in [None,{'cost':{}},{'cost':{(0,1):math.inf,(0,2):math.nan}}]:
        out=annotate_route_distances(s,m,lambda _:([0]*6,0),lambda *args:tree)
        assert all(c['planned_path_length_m'] is None and c['planned_path_length_state']=='unavailable' and c['planned_path_length_unavailable_reason'] for c in out['candidates'])
    s['estimated_pose']['x_m']=math.nan
    def forbidden(*args):raise AssertionError('invalid input must not start search')
    out=annotate_route_distances(s,m,forbidden,forbidden)
    assert all(c['planned_path_length_m'] is None for c in out['candidates'])
    m,s=fixture();s['slam_map']['captured_sim_time_s']=1.
    assert annotate_route_distances(s,m,forbidden,forbidden)['candidates'][0]['planned_path_length_unavailable_reason']=='map_timestamp_mismatch'
