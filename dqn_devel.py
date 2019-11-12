
# TODO: Roll most of main loop into the step function of the agent for more modularity
# TODO: Add param to select device
# TODO: add config option to disable progress bar

import gym
gym.logger.set_level(40)

import argparse, os, random, tqdm
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import save_config, update_linear_schedule, create_directory, LinearSchedule, PiecewiseSchedule, ExponentialSchedule
from utils.wrappers import make_env_atari
from utils.hyperparameters import Config
from utils.plot import plot_reward

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--env-name', default='BreakoutNoFrameskip-v4',
					help='environment to train on (default: BreakoutNoFrameskip-v4)')
parser.add_argument('--seed', type=int, default=None, help='random seed. Note if seed is None then it will be randomly generated (default: None)')
parser.add_argument('--stack-frames', type=int, default=4,
					help='Number of frames to stack (default: 4)')
parser.add_argument('--adaptive-repeat', nargs='+', type=int, default=[4],
                    help='Possible action repeat values (default: [4])')
parser.add_argument('--sticky-actions', type=float, default=0.,
                    help='Sticky action probability. I.e. the probability that input is ignored and the previous action is repeated (default: 0.)')
parser.add_argument('--algo', default='dqn',
					help='algorithm to use: dqn')
parser.add_argument('--print-threshold', type=int, default=1000,
					help='print progress and plot every print-threshold timesteps (default: 1000)')
parser.add_argument('--save-threshold', type=int, default=10000,
					help='save nn params every save-threshold timesteps (default: 1000)')
parser.add_argument('--eps-start', type=float, default=1.0,
					help='starting value of epsilon (default: 1.0)')
parser.add_argument('--eps-end', nargs='+', type=float, default=[0.1, 0.01],
					help='ending value of epsilon for each part of the peicewise function (default: [0.1, 0.01])')
parser.add_argument('--eps-decay', nargs='+', type=float, default=[0.05, 1.0],
					help='Percent of training at which each eps-end value will be reached via linear decay\n. Choose \
                        a single value > 1.0 to switch to an exponential decay schedule (default: [0.05, 1.0])')
parser.add_argument('--lr', type=float, default=1e-4,
					help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tnet-update', type=int, default=4e4,
					help='Num Steps between target net updates (default: 40000)')
parser.add_argument('--replay-size', type=int, default=1e6,
					help='!!!WATCH YOUR RAM USAGE!!! Size of replay buffer (default: 1000000)')
parser.add_argument('--batch-size', type=int, default=32,
					help='Size of minibatches drawn from replay buffer (default: 32)')
parser.add_argument('--learn-start', type=int, default=8e4,
					help='tstep to start updating (default: 80000)')
parser.add_argument('--max-tsteps', type=int, default=2e7,
					help='Maximimum number of timsteps to train (default: 2e7)')
parser.add_argument('--nenvs', type=int, default=1,
					help='number of parallel environments executing (default: 1)')
parser.add_argument('--update-freq', type=int, default=4,
					help='frequency (tsteps) to perform updates (default: 4)')
parser.add_argument('--max-grad-norm', type=float, default= 40.0,
					help='max norm of gradients (default: 40.0)')
parser.add_argument('--adam-eps', type=float, default=1e-4,
					help='epsilon param of adam (default: 1e-4)')
parser.add_argument('--anneal-lr', action='store_true', default=False,
					help='anneal lr from start value to 0 throught training')
parser.add_argument('--inference', action='store_true', default=False,
					help='Inference saved model')
parser.add_argument('--render', action='store_true', default=False,
                    help='Render the inference epsiode (default: False')
parser.add_argument('--state-norm', type=float, default=255.0,
					help='Normalization constant for states. Set to None if normalization \
                        is handled elsewhere (wrappers) or unneeded (default: 255.0)')

def train(config):
    #make/clear directories for logging
    base_dir = os.path.join('./results/', config.algo, config.env_id)
    log_dir = os.path.join(base_dir, 'logs/')   
    model_dir = os.path.join(base_dir, 'saved_model/')
    tb_dir = os.path.join(base_dir, 'runs/')
    create_directory(base_dir)
    create_directory(log_dir)
    create_directory(model_dir)
    create_directory(tb_dir)

    #Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(base_dir, 'runs'))
    
    #save configuration for later reference
    save_config(config, base_dir)

    #set seeds
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    envs = [make_env_atari(config.env_id, config.seed, i, log_dir, stack_frames=config.stack_frames, adaptive_repeat=config.adaptive_repeat, sticky_actions=config.sticky_actions, clip_rewards=True) for i in range(config.num_envs)]
    envs = DummyVecEnv(envs) if len(envs) == 1 else SubprocVecEnv(envs)

    model = Model(static_policy=config.inference, env=envs, config=config, log_dir=base_dir, tb_writer=writer)
    
    # episode_rewards = np.zeros(config.num_envs)
    episode_rewards = 0
    last_100_rewards = deque(maxlen=100)

    if len(config.epsilon_final) == 1:
        if config.epsilon_decay[0] > 1.0:
            anneal_eps = ExponentialSchedule(config.epsilon_start, config.epsilon_final[0], config.epsilon_decay[0], config.max_tsteps)
        else:
            anneal_eps = LinearSchedule(config.epsilon_start, config.epsilon_final[0], config.epsilon_decay[0], config.max_tsteps)
    else:
        anneal_eps = PiecewiseSchedule(config.epsilon_start, config.epsilon_final, config.epsilon_decay, config.max_tsteps)

    start = timer()
    observations = envs.reset()

    progress = tqdm.tqdm(range(1, int(config.max_tsteps) + 1))
    progress.set_description("Updates %d, Tsteps %d, Time %.2f, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
        (0, 0 , 0, 0, 0.0, 0.0, 0.0, 0.0))
    for current_tstep in progress:
        
        if config.use_lr_schedule:
            update_linear_schedule(model.optimizer, current_tstep-1, config.max_tsteps, config.lr)
        
        epsilon = anneal_eps(current_tstep)
        writer.add_scalar('Policy/Epsilon', epsilon, current_tstep)

        actions = model.get_action(observations, epsilon)

        prev_observations=observations
        observations, rewards, dones, infos = envs.step(actions)

        model.update(prev_observations, actions, rewards, observations, dones.astype(int), current_tstep)
        
        episode_rewards += rewards
        
        for idx, done in enumerate(dones):
            if done:
                model.finish_nstep(idx)
                model.reset_hx(idx)

                writer.add_scalar('Performance/Agent Reward', episode_rewards[idx], current_tstep)
                episode_rewards[idx] = 0

                # NOTE: no need to reset env. Vec env handles it
        
        for info in infos:
            if 'episode' in info.keys():
                    last_100_rewards.append(info['episode']['r'])
                    writer.add_scalar('Performance/Environment Reward', info['episode']['r'], current_tstep)
                    writer.add_scalar('Performance/Episode Length', info['episode']['l'], current_tstep)
            
        if current_tstep % config.save_threshold == 0:
            model.save_w()

        if current_tstep % config.print_threshold == 0 and last_100_rewards:
            end = timer()
            # print("Updates {}, num timesteps {}, Time Elapsed {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}".
            #     format(int(np.max([(current_tstep-config.learn_start)/config.update_freq, 0])),
            #         current_tstep,
            #         str(timedelta(seconds=end-start)).split('.')[0],
            #         int(current_tstep*np.mean(config.adaptive_repeat) / (end - start)),
            #         np.mean(last_100_rewards),
            #         np.median(last_100_rewards),
            #         np.min(last_100_rewards),
            #         np.max(last_100_rewards)))
            progress.set_description("Upd. %d, Tsteps %d, Time %s, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
                (int(np.max([(current_tstep-config.learn_start)/config.update_freq, 0])),
                current_tstep,
                str(timedelta(seconds=end-start)).split('.')[0],
                int(current_tstep*config.num_envs*np.mean(config.adaptive_repeat) / (end - start)),
                np.mean(last_100_rewards),
                np.median(last_100_rewards),
                np.min(last_100_rewards),
                np.max(last_100_rewards))
            )
            

    end = timer()
    if(last_100_rewards):
        progress.set_description("Upd. %d, Tsteps %d, Time %s, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
            (int(np.max([(config.max_tsteps-config.learn_start)/config.update_freq, 0])),
            config.max_tsteps,
            str(timedelta(seconds=end-start)).split('.')[0],
            int(config.max_tsteps*np.mean(config.adaptive_repeat) / (end - start)),
            np.mean(last_100_rewards),
            np.median(last_100_rewards),
            np.min(last_100_rewards),
            np.max(last_100_rewards))
        )
        
    model.save_w()
    envs.close()

if __name__=='__main__':
    args = parser.parse_args()

    #Import Correct Model
    if args.algo == 'dqn':
        from agents.DQN import Model
    else:
        print("INVALID ALGORITHM. ABORT.")
        exit()

    #training params
    config = Config()

    config.algo = args.algo
    config.env_id = args.env_name
    config.seed = args.seed
    config.inference = args.inference
    config.sticky_actions = args.sticky_actions
    config.use_lr_schedule = args.anneal_lr
    config.print_threshold = args.print_threshold
    config.save_threshold = args.save_threshold
    config.render = args.render
    config.s_norm = args.state_norm

    #preprocessing
    config.stack_frames = args.stack_frames
    config.adaptive_repeat = args.adaptive_repeat #adaptive repeat

    #algorithm control
    # config.noisy_nets      = True
    # config.priority_replay = True


    #Multi-step returns
    # config.N_STEPS = 1

    #epsilon variables
    config.epsilon_start    = args.eps_start
    config.epsilon_final    = args.eps_end
    config.epsilon_decay    = args.eps_decay

    #misc agent variables
    config.lr    = args.lr
    config.gamma = args.gamma

    #memory
    config.TARGET_NET_UPDATE_FREQ = args.tnet_update
    config.EXP_REPLAY_SIZE        = args.replay_size
    config.BATCH_SIZE             = args.batch_size
    # config.priority_alpha       = 0.6
    # config.priority_beta_start  = 0.4
    # config.priority_beta_tsteps = 100000

    #Noisy Nets
    # config.sigma_init = 0.5

    #Learning control variables
    config.learn_start = args.learn_start
    config.max_tsteps  = args.max_tsteps
    config.update_freq = args.update_freq
    config.num_envs    = args.nenvs

    #adam params
    config.adam_eps = args.adam_eps
    config.grad_norm_max = args.max_grad_norm

    #Categorical Params
    # config.ATOMS = 51
    # config.V_MAX = 10
    # config.V_MIN = -10

    #Quantile Regression Parameters
    # config.QUANTILES = 51

    #DRQN Parameters
    # config.SEQUENCE_LENGTH = 8

    train(config)