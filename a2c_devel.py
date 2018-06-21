import gym
import numpy as np

from IPython.display import clear_output
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
#%matplotlib inline

from timeit import default_timer as timer
from datetime import timedelta
import math
from random import randint

from utils.wrappers import *
from utils.subprocess_vec_env import *
from utils.hyperparameters import Config
from agents.A2C import Model

config = Config()

#a2c control
config.num_agents=16
config.rollout=5

#algorithm control
config.USE_NOISY_NETS=False
config.USE_PRIORITY_REPLAY=False

#misc agent variables
config.GAMMA=0.99
config.LR=3e-4

#Noisy Nets
config.SIGMA_INIT=0.5

config.MAX_FRAMES=100000000

#Categorical Params
config.ATOMS = 51
config.V_MAX = 50
config.V_MIN = 0

#Quantile Regression Parameters
config.QUANTILES=21

#DRQN Parameters
config.SEQUENCE_LENGTH=8


def plot(frame_idx, rewards, losses, sigma, elapsed_time):
    '''clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
    plt.plot(rewards)
    if losses:
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
    if sigma:
        plt.subplot(133)
        plt.title('noisy param magnitude')
        plt.plot(sigma)
    plt.show()'''
    print('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))


if __name__=='__main__':
    def make_env_cartpole(env_name):
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk

    def make_env_atari(env_name):
        def _thunk():
            env = make_atari(env_name)
            env = wrap_deepmind(env, frame_stack=False)
            env = wrap_pytorch(env)
            return env

        return _thunk

    start=timer()

    envs = []
    seed = randint(0,100)
    #env_id = "PongNoFrameskip-v4"
    #envs = [make_env_atari(env_id) for i in range(config.num_agents)]
    env_id = "CartPole-v0"
    envs = [make_env_cartpole(env_id) for i in range(config.num_agents)]
    envs = SubprocVecEnv(envs)

    model = Model(env=envs, config=config)

    frame_idx = 1
    print_step = 10000
    episode_reward = np.zeros(config.num_agents)
    rollout_mem = []

    observations = envs.reset()
    while frame_idx < config.MAX_FRAMES+1:
        for i in range(config.rollout):
            mask = np.ones(config.num_agents)
            policy, value = model.get_action(observations)

            actions = policy.multinomial(1).cpu().numpy().ravel()

            prev_observations=observations
            observations, rewards, dones, _ = envs.step(actions)
            episode_reward += rewards
            for j in range(config.num_agents):
                if dones[j]:
                    #no reset needed?
                    mask[j] = 0.0
                    model.save_reward(episode_reward[j])
                    episode_reward[j] = 0.0
            
            frame_idx+=config.num_agents
            rollout_mem.append((rewards, actions, mask, policy, value))

        _, value = model.get_action(observations)
        rollout_mem.append((None, None, None, None, value))
            
        model.update(rollout_mem)
        rollout_mem=[]
                
        if frame_idx > print_step:
            plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))
            print_step += 10000

    model.save_w()
    envs.close()