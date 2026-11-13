"""Minimal Cartesian observation/reset/step adapter, without a training reward."""
import numpy as np
from environment import Environment, cartesian_step


class CartesianInterface:
    def __init__(self):
        self.env=Environment()
        self.observation=None

    def reset(self, object_xy=(.22,.024), seed=0):
        self.observation=self.env.reset(object_xy=object_xy,seed=seed)
        return self.observation

    def step(self, action):
        """action: xyz displacement in metres per .1 s plus gripper target radians."""
        action=np.asarray(action,float)
        if action.shape!=(4,) or not np.isfinite(action).all():
            raise ValueError('expected four finite action values')
        if self.observation is None: raise RuntimeError('reset before step')
        self.observation=cartesian_step(self.env,self.observation,action[:3],action[3])
        # Learning code gets a camera frame every step, independent of logging cadence.
        self.observation=self.env.observe()
        return self.observation

    def close(self):
        self.env.close()
