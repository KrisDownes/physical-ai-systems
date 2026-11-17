"""The diagnostic must isolate rollouts and copy all MuJoCo data."""
import unittest
from pathlib import Path
import numpy as np
from decision_audit import audit,integration_state,mirror
from shared_evaluate import setup
from shared_tasks import SharedTask

class CandidateAuditTest(unittest.TestCase):
 def test_copy_and_selected_rollout_match_without_mutating_source(self):
  cond={'friction':.75,'mass_scale':1.25,'resistance':.01};start={'xy':[.217,.027],'pan':-.54}
  source=SharedTask('pull',**cond);clone=SharedTask('pull',**cond)
  try:
   f,grip,setting,_=setup(source,'pull',start,5401);clone.reset(start['xy'],5401,start['pan']);goal,_=setting
   mirror(source,clone);np.testing.assert_array_equal(integration_state(source),integration_state(clone))
   root=Path('results/so101_runs/20261116T220051.143395Z_aa753c5c56e8_shared_train_58a691af');weights={}
   for kind in ('current','history'):
    p=np.load(root/f'{kind}_seed1.npz');weights[kind]=[p[f'p{i}'] for i in range(4)]
   previous=integration_state(source);result=audit(source,clone,f,goal,grip,[],weights)
   np.testing.assert_array_equal(previous,integration_state(source));self.assertTrue(result['source_unchanged'])
   nxt=source.step([0.,-.006],grip);np.testing.assert_allclose(nxt,result['candidates'][1]['next_features'],atol=1e-6)
   self.assertFalse(result['candidates'][1]['numerical_invalid'])
  finally:source.close();clone.close()
if __name__=='__main__':unittest.main()
