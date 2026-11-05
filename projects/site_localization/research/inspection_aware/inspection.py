"""Camera-belief search arm; the frozen inspector handles all other phases."""
import math
from pathlib import Path
import sys

import rclpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'continuous_asset'))
from inspection import SearchMission
from belief_search import CameraBeliefSearch


class BeliefMission(SearchMission):
    def __init__(self, out):
        super().__init__(out)
        assert self.settings.get('search_selector') == 'camera_belief'
        self.belief_search = CameraBeliefSearch(self.planner, out/'map.txt', self.settings['search_coordinates'])

    def next_search(self):
        if self.vision_records:
            row = self.vision_records[-1]
            if row['purpose'] in ('old_view', 'search'):
                updated = self.belief_search.update_negative(row)
                if updated is not None:
                    self.record('belief_update', **updated)
        self.heading_index += 1
        if self.heading_index < 8:
            goal = [*self.locations[-1], self.heading_index*math.pi/4]
        else:
            if len(self.locations) >= self.settings['max_search_positions']:
                self.finish(False, 'search_budget_exhausted'); return
            goal, audit = self.belief_search.choose(
                self.pose[:2], self.locations, self.path_length, self.settings['max_path_m'])
            self.record('search_choice', visited=self.locations,
                        current_pose=self.pose.tolist(), **audit)
            if goal is None:
                self.finish(False, 'search_budget_exhausted_no_feasible_view'); return
            self.locations.append(goal[:2]); self.heading_index = 0
        if not self.navigate(goal, 'search'):
            self.finish(False, 'navigation_unreachable_search_goal')


if __name__ == '__main__':
    rclpy.init(); node = BeliefMission(Path(sys.argv[1]))
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.close(); node.destroy_node(); rclpy.try_shutdown()
