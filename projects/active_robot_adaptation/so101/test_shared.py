"""Shared interface, gradient, missingness, and condition split checks."""
import json,unittest
import numpy as np
from shared_tasks import SharedTask
from shared_train import gradient,predict,build

class SharedTests(unittest.TestCase):
 def test_gradient_multiple_parameters(self):
  rng=np.random.default_rng(22);x=rng.normal(size=(5,4)).astype(np.float32);y=rng.normal(size=(5,2)).astype(np.float32)
  w=[rng.normal(size=(4,6)).astype(np.float32),rng.normal(size=6).astype(np.float32),rng.normal(size=(6,2)).astype(np.float32),rng.normal(size=2).astype(np.float32)]
  _,g=gradient(w,x,y)
  for k,idx in [(0,(1,2)),(1,(1,)),(2,(3,1)),(3,(0,))]:
   old=w[k][idx];w[k][idx]=old+.001;a=np.mean((predict(w,x)-y)**2);w[k][idx]=old-.001;b=np.mean((predict(w,x)-y)**2);w[k][idx]=old
   self.assertLess(abs(float((a-b)/.002-g[k][idx])),.002)
 def test_hidden_parameters_and_visibility(self):
  for task in ('push','pull'):
   e=SharedTask(task,friction=.8,mass_scale=1.3,resistance=.03)
   try:
    f=e.reset();self.assertEqual(f.shape,(13,));self.assertEqual(f[-1],1.);self.assertNotIn('image',e.env.trace)
    self.assertAlmostEqual(e.env.model.body_mass[e.env.object_id],e.env.model.body_subtreemass[e.env.object_id])
    obs=e.env.observe();obs['image'][:]=0;missing=e.features(obs);self.assertEqual(missing[-1],0.);np.testing.assert_array_equal(missing[:4],0.)
   finally:e.close()
 def test_train_test_conditions_and_starts_disjoint(self):
  with open('so101/shared_protocol.json') as f:c=json.load(f)
  self.assertFalse(any(a==b for a in c['train_conditions'] for b in c['test_conditions']))
  self.assertFalse(any(a==b for a in c['train_starts'] for b in c['test_starts']))
 def test_missing_visual_target_is_not_train_data(self):
  obs=np.zeros(13);obs[-1]=1;nxt=obs.copy();nxt[-1]=0
  episode={'split':'train','transitions':[{'observation':obs,'action':[0,0],'next_observation':nxt}]}
  x,y=build([episode],'train',True);self.assertEqual(len(x),0);self.assertEqual(len(y),0)
if __name__=='__main__':unittest.main()
