#TODO: include wrapper to log original returns
#TODO: roll in loop stuff to step function so the same drive can be used for a2c/dqn algs


import os
import numpy as np
from collections import deque

import gym
from gym import spaces
from gym import wrappers
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import NoopResetEnv, TimeLimit, make_atari, wrap_deepmind
from baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, ClipRewardEnv, FrameStack

class MaxAndSkipEnv_custom(gym.Wrapper):
    def __init__(self, env, skip=[4], sticky_actions=0.0):
        """
        Return only every `skip`-th frame
        Adds support for adaptive repeat and sticky actions
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

        self.num_actions = env.action_space.n
        self.sticky = sticky_actions
        self.prev_action = None

    def step(self, a):
        """Repeat action, sum reward, and max over last observations."""
        repeat_len = a // self.num_actions
        action = a % self.num_actions

        total_reward = 0.0
        done = None
        for i in range(self._skip[repeat_len]):
            is_sticky = np.random.rand()
            if is_sticky >= self.sticky or self.prev_action is None:
                self.prev_action = action

            obs, reward, done, info = self.env.step(self.prev_action)
            if i == self._skip[repeat_len] - 2: self._obs_buffer[0] = obs
            if i == self._skip[repeat_len] - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_atari_custom(env_id, max_episode_steps=None, skip=[4], sticky_actions=0.0):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv_custom(env, skip, sticky_actions)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_custom(env, episode_life=True, clip_rewards=True, frame_stack=4, scale=False):
    """
    Configure environment for DeepMind-style Atari.
    Adds support for custom # of frames stacked
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack>1:
        env = FrameStack(env, frame_stack)
    return env

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
        #return observation.transpose(2, 0, 1)
        return np.array(observation).transpose(2, 0, 1)


def make_env_atari(env_id, seed, rank, log_dir, stack_frames=4, adaptive_repeat=[4], sticky_actions=0.0, clip_rewards=True):
    def _thunk():        
        env = make_atari_custom(env_id, max_episode_steps=None, skip=adaptive_repeat, sticky_actions=0.0)
        
        if seed:
            env.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind_custom(env, episode_life=True, clip_rewards=clip_rewards, frame_stack=stack_frames, scale=False)
        #env = atari_stack_and_repeat(env, stack_frames, adaptive_repeat, sticky_actions)
        env = WrapPyTorch(env)

        return env
    return _thunk


# class atari_stack_and_repeat(gym.Wrapper):
#     def __init__(self, env, k, adaptive_repeat, sticky):
#         """Stack k last frames.

#         Returns lazy array, which is much more memory efficient.

#         See Also
#         --------
#         baselines.common.atari_wrappers.LazyFrames
#         """
#         gym.Wrapper.__init__(self, env)
#         self.k = k
#         self.adaptive_repeat = adaptive_repeat
#         self.num_actions = env.action_space.n
#         self.frames = deque([], maxlen=k)
#         self.sticky = sticky
#         self.prev_action = None
#         shp = env.observation_space.shape
#         self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

#     def reset(self): #pylint: disable=method-hidden
#         ob = self.env.reset()
#         for _ in range(self.k):
#             self.frames.append(ob)
#         return self._get_ob()

#     def step(self, a): #pylint: disable=method-hidden
#         repeat_len = a // self.num_actions
#         action = a % self.num_actions

#         is_sticky = np.random.rand()
#         if is_sticky >= self.sticky or self.prev_action is None:
#             self.prev_action = action
#         ob, reward, done, info = self.env.step(self.prev_action)
        
#         self.frames.append(ob)
#         total_reward = reward
        
#         for i in range(1, self.adaptive_repeat[repeat_len]):
#             if not done:
#                 is_sticky = np.random.rand()
#                 if is_sticky >= self.sticky or self.prev_action is None:
#                     self.prev_action = action
#                 ob, reward, done, info = self.env.step(self.prev_action)

#                 total_reward += reward
#                 self.frames.append(ob)
#             else:
#                 self.frames.append(ob)
#         return self._get_ob(), total_reward, done, info

#     def _get_ob(self):
#         assert len(self.frames) == self.k
#         return LazyFrames(list(self.frames))