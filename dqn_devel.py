import gym
import numpy as np

from IPython.display import clear_output
import matplotlib
#matplotlib.use("agg")
from matplotlib import pyplot as plt
#%matplotlib inline

from timeit import default_timer as timer
from datetime import timedelta
import math

from utils.wrappers import *
from utils.hyperparameters import Config
from agents.DQN import Model

config = Config()

#algorithm control
config.USE_NOISY_NETS=False
config.USE_PRIORITY_REPLAY=False

#Multi-step returns
config.N_STEPS = 1

#epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 500
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA=0.99
config.LR=1e-4

#memory
config.TARGET_NET_UPDATE_FREQ = 128
config.EXP_REPLAY_SIZE = 10000
config.BATCH_SIZE = 32
config.PRIORITY_ALPHA=0.6
config.PRIORITY_BETA_START=0.4
config.PRIORITY_BETA_FRAMES = 100000

#Noisy Nets
config.SIGMA_INIT=0.5

#Learning control variables
config.LEARN_START = config.BATCH_SIZE*2
config.MAX_FRAMES=100000

#Categorical Params
config.ATOMS = 51
config.V_MAX = 50
config.V_MIN = 0

#Quantile Regression Parameters
config.QUANTILES=21

#DRQN Parameters
config.SEQUENCE_LENGTH=8


def plot(frame_idx, rewards, losses, sigma, elapsed_time):
    clear_output(True)
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
    plt.show()
    print('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))


if __name__=='__main__':
    start=timer()

    '''env_id = "PongNoFrameskip-v4"
    env    = make_atari(env_id)
    env    = wrap_deepmind(env, frame_stack=False)
    env    = wrap_pytorch(env)'''
    env = gym.make('CartPole-v0')
    #env = wrappers.Monitor(env, 'Delete', force=True)
    model = Model(env=env, config=config)

    episode_reward = 0

    observation = env.reset()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        epsilon = config.epsilon_by_frame(frame_idx)

        action = model.get_action(observation, epsilon)
        prev_observation=observation
        observation, reward, done, _ = env.step(action)
        observation = None if done else observation

        model.update(prev_observation, action, reward, observation, frame_idx)
        episode_reward += reward
        
        if done:
            model.finish_nstep()
            model.reset_hx()
            observation = env.reset()
            model.save_reward(episode_reward)
            episode_reward = 0
            
            
        if frame_idx % 10000 == 0:
            plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))

    model.save_w()
    env.close()