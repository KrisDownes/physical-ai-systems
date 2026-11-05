"""Consequential judge checks: repeated inspections cannot imply dock return."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from score import main as score


class DockContract(unittest.TestCase):
    def evaluate(self,return_dock=True,late=False):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            goals=[[1.,0.,0.],[2.,0.,0.],[1.,0.,0.]]
            cfg=dict(goals=goals,scenario='synthetic repeated inspections',dwell_s=2,
                judge_position_m=.15,judge_yaw_rad=.1745,max_sim_s=30,obstacles=[[10,10,1,1]],
                condition='unchanged',remedy='test',localizer='test',mapping='off',
                map_from_mission=[0,0,0],operator_events=[],reset_count=1,initialization='synthetic',
                inspection_zones=['unchanged']*3,
                dock_completion=dict(pose=[0,0,0],position_tolerance_m=.15,
                    yaw_tolerance_rad=.1745,stationary_dwell_s=2,required_after_inspection_index=2))
            (out/'cell.json').write_text(json.dumps(cfg))
            rows=[];t=0.
            # Initial dock occupation must never count as return completion.
            points=[[0.,0.,0.],*goals]
            if return_dock:points.append([0.,0.,0.])
            for index,p in enumerate(points):
                if late and index==len(points)-1:t=31.
                for _ in range(65):
                    t+=.05
                    rows.extend([dict(type='truth',time=t,x=p[0],y=p[1],q=[0,0,0,1]),
                                 dict(type='wheel_contacts',time=t,count=1)])
            (out/'judge.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with contextlib.redirect_stdout(io.StringIO()):score(out)
            return json.loads((out/'outcome.json').read_text())

    def test_repeated_visits_plus_actual_dock(self):
        result=self.evaluate()
        self.assertEqual(result['physical_stops_completed'],3)
        self.assertTrue(result['mission_success'])

    def test_initial_dock_does_not_count_as_return(self):
        result=self.evaluate(return_dock=False)
        self.assertEqual(result['physical_stops_completed'],3)
        self.assertFalse(result['dock_return_success'])
        self.assertFalse(result['mission_success'])

    def test_late_return_fails_even_if_inspections_pass(self):
        result=self.evaluate(late=True)
        self.assertTrue(result['inspection_success'])
        self.assertFalse(result['mission_success'])


if __name__=='__main__':unittest.main()
