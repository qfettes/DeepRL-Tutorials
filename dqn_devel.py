import gym
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
import math
import glob

from utils.wrappers import *
from utils.hyperparameters import Config
from agents.DQN import Model
from utils.plot import plot_all_data

config = Config()

#algorithm control
config.USE_NOISY_NETS      = False
config.USE_PRIORITY_REPLAY = False

#Multi-step returns
config.N_STEPS = 1

#epsilon variables
config.epsilon_start    = 1.0
config.epsilon_final    = 0.01
config.epsilon_decay    = 30000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

#misc agent variables
config.GAMMA = 0.99
config.LR    = 1e-4

#memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE        = 100000
config.BATCH_SIZE             = 32

config.PRIORITY_ALPHA       = 0.6
config.PRIORITY_BETA_START  = 0.4
config.PRIORITY_BETA_FRAMES = 100000

#Noisy Nets
config.SIGMA_INIT = 0.5

#Learning control variables
config.LEARN_START = 10000
config.MAX_FRAMES  = 1000000
config.UPDATE_FREQ = 1

#Categorical Params
config.ATOMS = 51
config.V_MAX = 50
config.V_MIN = 0

#Quantile Regression Parameters
config.QUANTILES = 21

#DRQN Parameters
config.SEQUENCE_LENGTH = 8

#data logging parameters
config.ACTION_SELECTION_COUNT_FREQUENCY = 1000

if __name__=='__main__':
    start=timer()

    log_dir = "/tmp/gym/"
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv')) \
            + glob.glob(os.path.join(log_dir, '*td.csv')) \
            + glob.glob(os.path.join(log_dir, '*sig_param_mag.csv')) \
            + glob.glob(os.path.join(log_dir, '*action_log.csv'))
        for f in files:
            os.remove(f)

    env_id = "PongNoFrameskip-v4"
    env    = make_atari(env_id)
    env    = bench.Monitor(env, os.path.join(log_dir, env_id))
    env    = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=True)
    env    = WrapPyTorch(env)
    model  = Model(env=env, config=config, log_dir=log_dir)

    episode_reward = 0

    observation = env.reset()
    for frame_idx in range(1, config.MAX_FRAMES + 1):
        epsilon = config.epsilon_by_frame(frame_idx)

        action = model.get_action(observation, epsilon)
        model.save_action(action, frame_idx) #log action selection

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
            model.save_w()
            try:
                print('frame %s. time: %s' % (frame_idx, timedelta(seconds=int(timer()-start))))
                plot_all_data(log_dir, env_id, 'DRQN', config.MAX_FRAMES, bin_size=(10, 100, 100, 1), smooth=1, time=timedelta(seconds=int(timer()-start)), ipynb=False)
            except IOError:
                pass

    model.save_w()
    env.close()