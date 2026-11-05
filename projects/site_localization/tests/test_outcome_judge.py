"""A completed command sequence must not substitute for physical completion."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('score_mission',Path(__file__).resolve().parents[1]/'sim/score_mission.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class OutcomeJudge(unittest.TestCase):
    def fixture(self,path,*,offset=0,contact=False,heartbeat=True):
        cfg=dict(goals=[[1,2,0]],dwell_s=2,judge_position_m=.15,judge_yaw_rad=.17,
                 max_sim_s=150,obstacles=[[5,5,1,1]],scenario='test')
        (path/'cell.json').write_text(json.dumps(cfg))
        rows=[]
        for i in range(70):
            t=i*.05
            rows.append(dict(type='truth',time=t,x=1+offset,y=2,q=[0,0,0,1]))
            if heartbeat:rows.append(dict(type='wheel_contacts',time=t,count=1))
        if contact:rows.append(dict(type='contacts',time=2,count=1))
        rows.append(dict(type='event',event=dict(type='controller_complete',time=3)))
        (path/'judge.jsonl').write_text('\n'.join(map(json.dumps,rows)))

    def check(self,**kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.fixture(path,**kwargs);return module.score(path)

    def test_physical_dwell_and_live_contacts_pass(self):self.assertTrue(self.check()['mission_success'])
    def test_claimed_completion_at_wrong_pose_fails(self):self.assertFalse(self.check(offset=.3)['mission_success'])
    def test_body_collision_fails_despite_completed_stops(self):self.assertFalse(self.check(contact=True)['mission_success'])
    def test_missing_contact_liveness_cannot_certify(self):self.assertFalse(self.check(heartbeat=False)['mission_success'])


if __name__=='__main__':unittest.main()
