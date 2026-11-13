"""Validate autodiff and a Gaussian policy-gradient update on a known bandit."""
import json
import time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, nn

start = time.perf_counter()
x = Tensor([3.])
(x*x).sum().backward()
gradient = float(x.grad.numpy()[0])
assert abs(gradient-6.) < 1e-5

rng = np.random.default_rng(21)
mu = Tensor([0.])
opt = nn.optim.Adam([mu], lr=.03)
initial = float(mu.numpy()[0])
for _ in range(100):
    center = float(mu.numpy()[0])
    action = rng.normal(center, .5, 256).astype(np.float32)
    reward = -(action-1.)**2
    advantage = reward-reward.mean()
    log_prob = -((Tensor(action)-mu)**2)/(2*.5**2)
    loss = -(log_prob * Tensor(advantage)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
final = float(mu.numpy()[0])
assert abs(final-1.) < .2, final
result = {'gradient_x_squared_at_3': gradient, 'initial_mean': initial, 'final_mean': final,
          'steps': 100, 'batch_size': 256, 'wall_s': time.perf_counter()-start,
          'seed': 21, 'task': 'one-step Gaussian bandit, optimum mean 1'}
path = Path('results/arm_transfer/tinygrad_sanity.json')
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
