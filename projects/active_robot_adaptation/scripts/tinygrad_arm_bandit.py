"""Tinygrad contextual-bandit policy for nominal 0.1 s arm displacement."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, nn

OUT = Path('results/arm_transfer/tinygrad')


def tip(q):
    return np.stack([.3*np.cos(q[:, 0])+.2*np.cos(q[:, 0]+q[:, 1]),
                     .3*np.sin(q[:, 0])+.2*np.sin(q[:, 0]+q[:, 1])], axis=1)


class Policy:
    def __init__(self, rng):
        self.w1 = Tensor((rng.normal(0, .25, (4, 24))).astype(np.float32))
        self.b1 = Tensor(np.zeros(24, np.float32))
        self.w2 = Tensor((rng.normal(0, .05, (24, 2))).astype(np.float32))
        self.b2 = Tensor(np.zeros(2, np.float32))
        self.params = [self.w1, self.b1, self.w2, self.b2]

    def __call__(self, features):
        return (((features @ self.w1 + self.b1).tanh() @ self.w2 + self.b2).tanh()) * .1


def batch(rng, n):
    q = np.column_stack((rng.uniform(-.9, .9, n), rng.uniform(-2., -.7, n)))
    obs = tip(q)
    desired = rng.uniform(-.025, .025, (n, 2))
    norm = np.linalg.norm(desired, axis=1, keepdims=True)
    desired *= np.minimum(1., .025/np.maximum(norm, 1e-10))
    features = np.column_stack((obs/.5, desired/.025)).astype(np.float32)
    return q, obs, desired, features


def train(seed, updates, n, save):
    rng = np.random.default_rng(seed)
    policy = Policy(rng)
    optimizer = nn.optim.Adam(policy.params, lr=.003)
    started = time.perf_counter()
    rewards = []
    for step in range(updates):
        q, obs, desired, features = batch(rng, n)
        mean = policy(Tensor(features))
        mean_value = mean.numpy()
        actions = mean_value + rng.normal(0., .035, (n, 2)).astype(np.float32)
        executed = np.clip(actions, -.1, .1) * .8
        actual = tip(q + executed) - obs
        reward = -np.sum(((actual-desired)/.025)**2, axis=1)
        advantage = (reward-reward.mean()).astype(np.float32)
        logprob = -((Tensor(actions)-mean)**2).sum(axis=1)/(2*.035**2)
        loss = -(logprob * Tensor(advantage)).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        rewards.append(float(reward.mean()))
    elapsed = time.perf_counter()-started
    data = {'seed': seed, 'updates': updates, 'batch_size': n, 'wall_s': elapsed,
            'initial_reward': float(np.mean(rewards[:min(5, updates)])),
            'final_reward': float(np.mean(rewards[-min(5, updates):]))}
    if save:
        np.savez(OUT / f'policy_seed_{seed}.npz', **{f'p{i}': p.numpy() for i,p in enumerate(policy.params)})
        (OUT / f'train_seed_{seed}.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(data), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        train(seed, args.updates, args.batch, args.save)
