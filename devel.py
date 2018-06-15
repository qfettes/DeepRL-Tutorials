import gym
import numpy as np

from IPython.display import clear_output
from matplotlib import pyplot as plt
#%matplotlib inline

from timeit import default_timer as timer
from datetime import timedelta

from utils.wrappers import *
from utils.hyperparameters import *
from agents.Categorical_DQN import Model


def plot(frame_idx, rewards, losses, elapsed_time):
    #clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
    #print('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))


if __name__=='__main__':
    start=timer()

    '''env_id = "PongNoFrameskip-v4"
    env    = make_atari(env_id)
    env    = wrap_deepmind(env, frame_stack=False)
    env    = wrap_pytorch(env)'''
    env = gym.make('CartPole-v0')
    #env = wrappers.Monitor(env, 'Delete', force=True)
    model = Model(env=env)

    losses = []
    all_rewards = []
    episode_reward = 0

    observation = env.reset()
    for frame_idx in range(1, MAX_FRAMES + 1):
        epsilon = epsilon_by_frame(frame_idx)

        action = model.get_action(observation, epsilon)
        prev_observation=observation
        observation, reward, done, _ = env.step(action)
        observation = None if done else observation

        loss = model.update(prev_observation, action, reward, observation, frame_idx)
        episode_reward += reward
        
        if done:
            model.finish_nstep()
            observation = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
            
        if loss is not None:
            losses.append(loss)
            
        if frame_idx % 10000 == 0:
            plot(frame_idx, all_rewards, losses, timedelta(seconds=int(timer()-start)))

    env.close()