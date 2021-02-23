from __future__ import absolute_import

import torch
import gym
from agents.SAC import Agent as Agent
from utils.hyperparameters import Config
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import pybulletgym
import os
import numpy as np
from itertools import chain
import random
from utils.wrappers import make_one_continuous
from utils import create_directory

gym.logger.set_level(40)

# declare_networks is fine with no test
# update target net is tested by update_

def test_append_to_replay():
    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    config = Config()
    config.device = 'cpu'
    config.inference = True
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

    assert(np.allclose(o0, state[0], atol=1e-4) or 
        np.allclose(o0, state[1], atol=1e-4) or 
        np.allclose(o1, state[0], atol=1e-4) or 
        np.allclose(o1, state[1], atol=1e-4))

    assert(np.allclose(o1, non_final_next_state[0], atol=1e-4) or 
        np.allclose(o1, non_final_next_state[1], atol=1e-4) or 
        np.allclose(o2, non_final_next_state[0], atol=1e-4) or 
        np.allclose(o2, non_final_next_state[1], atol=1e-4))

    assert(np.allclose(r1, reward[0], atol=1e-4) or 
        np.allclose(r1, reward[1], atol=1e-4) or 
        np.allclose(r2, reward[0], atol=1e-4) or 
        np.allclose(r2, reward[1], atol=1e-4))

    env.close()
    
def test_prep_minibatch():
    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.batch_size = 1
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    o0 = env.reset()
    a0 = env.action_space.sample()
    o1, r1, d1, _ = env.step(a0)

    agent.append_to_replay(o0, a0.reshape(1, -1), r1, o1, d1)

    batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, _, _ = agent.prep_minibatch(1)

    assert(batch_state.shape == (1,)+env.observation_space.shape)
    assert(batch_action.shape == (1,)+env.action_space.shape)
    assert(batch_reward.shape == (1,1))
    assert(non_final_next_states.shape == (1,)+env.observation_space.shape)
    assert(non_final_mask.shape == (1,))

    assert(batch_state.dtype == torch.float)
    assert(batch_action.dtype == torch.float)
    assert(batch_reward.dtype == torch.float)
    assert(non_final_next_states.dtype == torch.float)
    assert(non_final_mask.dtype == torch.bool)

    batch_state = batch_state.cpu().numpy()
    batch_action = batch_action.cpu().numpy()
    batch_reward = batch_reward.cpu().numpy()
    non_final_next_states = non_final_next_states.cpu().numpy()
    non_final_mask = non_final_mask.cpu().numpy()

    assert(np.allclose(batch_state * config.state_norm, o0, atol=1e-4))
    assert(np.allclose(batch_action, a0, atol=1e-4))
    assert(np.allclose(batch_reward, r1, atol=1e-4))
    assert(np.allclose(non_final_next_states * config.state_norm, o1, atol=1e-4))
    
    assert(all(non_final_mask))
    assert(not empty_next_state_values)

    env.close()

def test_compute_value_loss():
    #TODO also test priority replay here

    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.batch_size = 10
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    scale = 0.01

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())
    grad_params = agent.q_net.parameters()

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # generate deterministic dummy data
    batch_state = torch.ones((config.batch_size,)+obs_shape, device=agent.device, dtype=torch.float)
    batch_action = torch.ones((config.batch_size,)+act_shape, device=agent.device, dtype=torch.float)
    batch_reward = torch.ones((config.batch_size,1), device=agent.device, dtype=torch.float) 
    non_final_next_states = torch.ones((config.batch_size,)+obs_shape, device=agent.device, dtype=torch.float)
    non_final_mask = torch.ones((config.batch_size,), device=agent.device, dtype=torch.bool)
    empty_next_state_values = False

    # assure determinism
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    value_loss = agent.compute_value_loss(
        (
            batch_state, batch_action, batch_reward, non_final_next_states, 
            non_final_mask, empty_next_state_values, None, None,
        ),
        tstep=config.batch_size
    )

    assert(np.allclose(value_loss.item(), 0.3401326537132263, atol=1e-4))

    agent.value_optimizer.zero_grad()
    value_loss.backward()

    grad_norm = 0.
    for p in grad_params:
        param_norm = p.grad.data.norm(2)
        grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** (1./2.)

    print(grad_norm)
    assert(np.allclose(grad_norm, 15.181056289401262, atol=1e-4))

    env.close()

def test_compute_policy_loss():
    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    obs_shape = env.observation_space.shape
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.batch_size = 10
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    scale = 0.01

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())
    grad_params = agent.policy_net.parameters()

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # generate deterministic dummy data
    batch_state = torch.ones((config.batch_size,)+obs_shape, device=agent.device, dtype=torch.float)

    # assure determinism
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    policy_loss, _ = agent.compute_policy_loss(
        (
            batch_state, None, None, None, 
            None, None, None, None,
        ),
        tstep=config.batch_size
    )

    assert(np.allclose(policy_loss.item(), -1.5993025302886963, atol=1e-4))

    agent.value_optimizer.zero_grad()
    policy_loss.backward()

    grad_norm = 0.
    for p in grad_params:
        param_norm = p.grad.data.norm(2)
        grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** (1./2.)

    assert(np.allclose(grad_norm, 2.86652455897556, atol=1e-4))

    env.close()

def test_compute_entropy_loss():
    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    act_shape = env.action_space.shape
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.batch_size = 10
    config.entropy_tuning = True
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    scale = 0.01

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())
    grad_params = [agent.log_entropy_coef]

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # generate deterministic dummy data
    action_log_probs = torch.ones((config.batch_size,1), device=agent.device, dtype=torch.float) / act_shape[0]
    action_log_probs = action_log_probs.log()

    # assure determinism
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # we want a non-zero entropy coef to verify
    # do one step
    entropy_loss = agent.compute_entropy_loss(
        action_log_probs,
        tstep=config.batch_size
    )
    agent.entropy_optimizer.zero_grad()
    entropy_loss.backward()
    agent.entropy_optimizer.step()

    # real step for verification
    entropy_loss = agent.compute_entropy_loss(
        action_log_probs,
        tstep=config.batch_size
    )
    agent.entropy_optimizer.zero_grad()
    entropy_loss.backward()

    assert(np.allclose(entropy_loss.item(), -0.0007791760144755244, atol=1e-4))

    grad_norm = 0.
    for p in grad_params:
        param_norm = p.grad.data.norm(2)
        grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** (1./2.)

    assert(np.allclose(grad_norm, 7.7917585372924805, atol=1e-4))

    env.close()

def test_compute_loss():
    env = DummyVecEnv([lambda: gym.make("HalfCheetahPyBulletEnv-v0")])
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.batch_size = 10
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    scale = 0.01

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())
    grad_params = agent.q_net.parameters()

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # generate deterministic dummy data
    batch_state = torch.ones((config.batch_size,)+obs_shape, device=agent.device, dtype=torch.float)
    batch_action = torch.ones((config.batch_size,)+act_shape, device=agent.device, dtype=torch.float)
    batch_reward = torch.ones((config.batch_size,1), device=agent.device, dtype=torch.float) 
    non_final_next_states = torch.ones((config.batch_size,)+obs_shape, device=agent.device, dtype=torch.float)
    non_final_mask = torch.ones((config.batch_size,), device=agent.device, dtype=torch.bool)
    empty_next_state_values = False

    # assure determinism
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    loss = agent.compute_loss(
        (
            batch_state, batch_action, batch_reward, non_final_next_states, 
            non_final_mask, empty_next_state_values, None, None,
        ),
        tstep=config.batch_size
    )

    assert(np.allclose(loss, -1.3114541172981262, atol=1e-4))

    env.close()

def test_update_():
    env = DummyVecEnv([make_one_continuous('HalfCheetahPyBulletEnv-v0', 0, 0, None)])
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.seed = 0
    config.batch_size = 10
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())

    scale = 0.01

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # generate some fake data
    o = np.ones((1,)+obs_shape, dtype=float) * scale
    for i in range(config.batch_size):
        a = np.ones((1,)+act_shape, dtype=np.float32) * scale
        r = np.ones((1,), dtype=float) * scale
        o2 = (np.ones((1,)+obs_shape, dtype=float) + i) * scale
        d = np.ones((1,), dtype=bool) * scale

        agent.append_to_replay(o, a.reshape(1, -1), r, o2, d)

        o = o2  

    # assure determinism in action selection
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    agent.update_(
        tstep=config.batch_size
    )

    value_param_norm = 0.
    for p in agent.q_net.parameters():
        param_norm = p.data.norm(2)
        value_param_norm += param_norm.item() ** 2
    value_param_norm = value_param_norm ** (1./2.)

    assert(np.allclose(value_param_norm, 3.8841390192814185, atol=1e-4))

    policy_param_norm = 0.
    for p in agent.policy_net.parameters():
        param_norm = p.data.norm(2)
        policy_param_norm += param_norm.item() ** 2
    policy_param_norm = policy_param_norm ** (1./2.)

    assert(np.allclose(policy_param_norm, 2.7679626595836915, atol=1e-4))

    target_value_param_norm = 0.
    for p in agent.target_q_net.parameters():
        param_norm = p.data.norm(2)
        target_value_param_norm += param_norm.item() ** 2
    target_value_param_norm = target_value_param_norm ** (1./2.)

    assert(np.allclose(target_value_param_norm, 3.8594992527502163, atol=1e-4))

    env.close()

def test_get_action():
    env = DummyVecEnv([make_one_continuous('HalfCheetahPyBulletEnv-v0', 0, 0, None)])
    obs_shape = env.observation_space.shape
    config = Config()
    config.device = 'cpu'
    config.seed = 0
    config.batch_size = 10
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())

    scale = 0.01

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # generate some fake data
    o = np.ones((1,)+obs_shape, dtype=float) * scale

    # assure determinism in action selection
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    action = agent.get_action(
        o,
        deterministic=False
    )
    expected_action = np.array([0.9474563,  -0.21982896, -0.9804724,  0.6231712,  -0.8000756,  -0.89496773])
    assert(np.allclose(action, expected_action, atol=1e-4))

    # check deteriminsitic action selection
    action = agent.get_action(
        o,
        deterministic=True
    )
    expected_action = np.array([0.10085898, 0.10085898, 0.10085898, 0.10085898, 0.10085898, 0.10085898])
    assert(np.allclose(action, expected_action, atol=1e-4))

    env.close()


def test_step():
    env = DummyVecEnv([make_one_continuous('HalfCheetahPyBulletEnv-v0', 0, 0, None)])
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.seed = 0
    config.batch_size = 10
    var_config = vars(config)

    agent = Agent(env, config, log_dir=None, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())

    scale = 0.01

    #fill parameters of model for consistent testing
    for p in all_params:
        p.data.fill_(scale)

    # assure determinism in action selection
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    agent.step(0, 0)

    assert(len(agent.memory)==1)

    env.close()

def test_load_save():
    env = DummyVecEnv([make_one_continuous('HalfCheetahPyBulletEnv-v0', 0, 0, None)])
    config = Config()
    config.device = 'cpu'
    config.inference = True
    config.seed = 0
    config.batch_size = 10
    config.entropy_tuning = True
    config.logdir='./unittests/'
    create_directory(os.path.join(config.logdir, 'saved_model'))
    var_config = vars(config)

    agent = Agent(env, config, log_dir=config.logdir, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )
    agent.log_entropy_coef=torch.tensor([1.0], dtype=torch.float, device=agent.device, requires_grad=True)

    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())

    pre_save_norm = 0.
    for p in list(all_params) + [agent.log_entropy_coef]:
        param_norm = p.data.norm(2)
        pre_save_norm += param_norm.item() ** 2
    pre_save_norm = pre_save_norm ** (1./2.)

    agent.save_w()
    del(agent)

    agent = Agent(env, config, log_dir=config.logdir, tb_writer=None, 
        valid_arguments=set(var_config.keys()), 
        default_arguments=var_config
    )
    agent.load_w()
    all_params = chain(agent.policy_net.parameters(), agent.q_net.parameters(), agent.target_q_net.parameters())

    post_save_norm = 0.
    for p in list(all_params) + [agent.log_entropy_coef]:
        param_norm = p.data.norm(2)
        post_save_norm += param_norm.item() ** 2
    post_save_norm = post_save_norm ** (1./2.)

    assert(np.allclose(pre_save_norm, post_save_norm, atol=1e-4))

    env.close()