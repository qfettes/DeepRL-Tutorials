from __future__ import absolute_import

import gym
from agents.SAC import Agent as Agent
from utils.hyperparameters import Config
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import pybulletgym
import os
import numpy as np

gym.logger.set_level(40)

# declare_networks is fine with no test

# prep_minibatch

def test_append_to_replay():
    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    config = Config()
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    o0 = env.reset()
    a0 = env.action_space.sample()
    o1, r1, d1, _ = env.step(a0)

    agent.append_to_replay(o0, a0.reshape(1, -1), r1, o1, d1)

    a1 = env.action_space.sample()
    o2, r2, d2, _ = env.step(a1)

    agent.append_to_replay(o1, a1.reshape(1, -1), r2, o2, d2)

    (state, action, reward, non_final_next_state, non_final_mask, empty_next_state), _, _ = agent.memory.sample(2)

    assert(state.shape == (2,)+env.observation_space.shape)
    assert(action.shape == (2,)+env.action_space.shape)
    assert(reward.shape == (2,))
    assert(non_final_next_state.shape == (2,)+env.observation_space.shape)
    assert(non_final_mask.shape == (2,))
    assert(empty_next_state == False)

    print(np.allclose(o0, state[0]))
    assert(np.allclose(o0, state[0]) or 
        np.allclose(o0, state[1]) or 
        np.allclose(o1, state[0]) or 
        np.allclose(o1, state[1]))

    assert(np.allclose(o1, non_final_next_state[0]) or 
        np.allclose(o1, non_final_next_state[1]) or 
        np.allclose(o2, non_final_next_state[0]) or 
        np.allclose(o2, non_final_next_state[1]))

    assert(np.allclose(r1, reward[0]) or 
        np.allclose(r1, reward[1]) or 
        np.allclose(r2, reward[0]) or 
        np.allclose(r2, reward[1]))

    env.close()
    

def test_dummy():
    assert True