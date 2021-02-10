# TODO: include wrapper to log original returns

import os
from collections import deque

import gym
import numpy as np
import torch
from baselines import bench
from baselines.common.atari_wrappers import (ClipRewardEnv, EpisodicLifeEnv,
                                             FireResetEnv, FrameStack,
                                             NoopResetEnv, ScaledFloatFrame,
                                             TimeLimit, WarpFrame, make_atari,
                                             wrap_deepmind)
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
from gym import spaces, wrappers
from gym.spaces.box import Box


class MaxAndSkipEnv_custom(gym.Wrapper):
    def __init__(self, env, skip=[4], sticky_actions=0.0):
        """
        Return only every `skip`-th frame
        Adds support for adaptive repeat and sticky actions
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

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
            if i == self._skip[repeat_len] - 2:
                self._obs_buffer[0] = obs
            if i == self._skip[repeat_len] - 1:
                self._obs_buffer[1] = obs
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
    if frame_stack > 1:
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
        # return observation.transpose(2, 0, 1)
        return np.array(observation).transpose(2, 0, 1)


def make_envs_general(env_id, seed, log_dir, num_envs, stack_frames=4, adaptive_repeat=[4], sticky_actions=0.0, clip_rewards=True):
    env = gym.make(env_id)
    atari = True if env.action_space.__class__.__name__ == 'Discrete' else False
    env.close()

    if atari:
        return make_all_atari(env_id, seed, log_dir, num_envs, stack_frames, adaptive_repeat, sticky_actions, clip_rewards)
    else:
        return make_all_continuous(env_id, seed, log_dir, num_envs)


def make_all_atari(env_id, seed, log_dir, num_envs, stack_frames=4, adaptive_repeat=[4], sticky_actions=0.0, clip_rewards=True):
    envs = [make_one_atari(env_id, seed, i, log_dir, stack_frames=stack_frames, adaptive_repeat=adaptive_repeat,
                           sticky_actions=sticky_actions, clip_rewards=True) for i in range(num_envs)]
    envs = DummyVecEnv(envs) if len(envs) == 1 else SubprocVecEnv(envs)

    return envs


def make_one_atari(env_id, seed, rank, log_dir, stack_frames=4, adaptive_repeat=[4], sticky_actions=0.0, clip_rewards=True):
    def _thunk():
        env = make_atari_custom(env_id, max_episode_steps=None,
                                skip=adaptive_repeat, sticky_actions=sticky_actions)

        if seed:
            env.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        env = wrap_deepmind_custom(
            env, episode_life=True, clip_rewards=clip_rewards, frame_stack=stack_frames, scale=False)
        env = WrapPyTorch(env)

        return env
    return _thunk


def make_all_continuous(env_id, seed, log_dir, num_envs):
    envs = [make_one_continuous(env_id, seed, i, log_dir)for i in range(num_envs)]
    envs = DummyVecEnv(envs) if len(envs) == 1 else SubprocVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, ret=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)

    # envs = VecPyTorch(envs, device)

    # if frame_stack is not None:
    #     envs = VecPyTorchFrameStack(envs, frame_stack, device)
    # elif len(envs.observation_space.shape) == 3:
    #     envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


def make_one_continuous(env_id, seed, rank, log_dir):
    def _thunk():
        env = gym.make(env_id)

        if seed:
            env.seed(seed + rank)
        else:
            env.seed(np.random.randint(10000000000))

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        return env
    return _thunk

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


# class VecNormalize(VecNormalize_):
#     def __init__(self, *args, **kwargs):
#         super(VecNormalize, self).__init__(*args, **kwargs)
#         self.training = True

#     def _obfilt(self, obs, update=True):
#         if self.ob_rms:
#             if self.training and update:
#                 self.ob_rms.update(obs)
#             obs = np.clip((obs - self.ob_rms.mean) /
#                           np.sqrt(self.ob_rms.var + self.epsilon),
#                           -self.clipob, self.clipob)
#             return obs
#         else:
#             return obs

#     def train(self):
#         self.training = True

#     def eval(self):
#         self.training = False

# # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# # Checks whether done was caused my timit limits or not


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# # Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# class VecPyTorch(VecEnvWrapper):
#     def __init__(self, venv, device):
#         """Return only every `skip`-th frame"""
#         super(VecPyTorch, self).__init__(venv)
#         self.device = device
#         # TODO: Fix data types

#     def reset(self):
#         obs = self.venv.reset()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         return obs

#     def step_async(self, actions):
#         if isinstance(actions, torch.LongTensor):
#             # Squeeze the dimension for discrete actions
#             actions = actions.squeeze(1)
#         actions = actions.cpu().numpy()
#         self.venv.step_async(actions)

#     def step_wait(self):
#         obs, reward, done, info = self.venv.step_wait()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
#         done = torch.from_numpy(done).unsqueeze(dim=1).to(torch.bool)
#         return obs, reward, done, info

# # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
# class VecPyTorchFrameStack(VecEnvWrapper):
#     def __init__(self, venv, nstack, device=None):
#         self.venv = venv
#         self.nstack = nstack

#         wos = venv.observation_space  # wrapped ob space
#         self.shape_dim0 = wos.shape[0]

#         low = np.repeat(wos.low, self.nstack, axis=0)
#         high = np.repeat(wos.high, self.nstack, axis=0)

#         if device is None:
#             device = torch.device('cpu')
#         self.stacked_obs = torch.zeros((venv.num_envs, ) +
#                                        low.shape).to(device)

#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stacked_obs[:, :-self.shape_dim0] = \
#             self.stacked_obs[:, self.shape_dim0:]
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stacked_obs[i] = 0
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs, rews, news, infos

#     def reset(self):
#         obs = self.venv.reset()
#         if torch.backends.cudnn.deterministic:
#             self.stacked_obs = torch.zeros(self.stacked_obs.shape)
#         else:
#             self.stacked_obs.zero_()
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs

#     def close(self):
#         self.venv.close()
