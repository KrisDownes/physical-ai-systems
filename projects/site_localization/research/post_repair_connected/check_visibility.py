"""Development consistency check against frozen map-only diagnostic choices."""
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'research/moved_asset'))
sys.path.insert(0,str(root/'research/continuous_asset'))
from buffered_planning import BufferedPlanner
from visibility_search import VisibilitySearch

run=root/'results/post_repair_connected_20261027/mission_wrong_asset/inspection'
preview=json.loads((root/'results/post_repair_connected_20261027/search_visibility_preview_v2.json').read_text())
policy=json.loads((run/'public/policy.json').read_text())
planner=BufferedPlanner(run/'map.txt')
selector=VisibilitySearch(planner,run/'map.txt',policy['search_coordinates'])
for state in preview['states']:
    chosen,audit=selector.choose(state['start'],state['visited'])
    assert chosen[:2]==state['max_gain_per_effort'],(state['state'],chosen)
    expected={tuple(x['station']):x for x in state['eligible']}
    for item in audit['options']:
        other=expected[tuple(item['station'])]
        assert abs(item['new_visible_area_m2']-other['new_visible_area_m2'])<1e-5
        assert abs(item['path_m']-other['path_m'])<1e-9
    print(state['state'],chosen[:2],round(audit['elapsed_wall_s'],3),'s')
