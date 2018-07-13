import numpy as np

import gym
from gym import spaces
from gym.spaces.box import Box

import os

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)/255.0
    
def wrap_pytorch(env):
    return ImageToPyTorch(env)

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

def make_env_a2c_atari(env_id, seed, rank, log_dir):
    def _thunk():
        env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind(env)

        obs_shape = env.observation_space.shape
        env = WrapPyTorch(env)

        return env
    return _thunk