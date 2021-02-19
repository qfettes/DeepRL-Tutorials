
# TODO: add arg to control type of noise in noisy nets
# TODO: efficiency for priority replay functions
# TODO: add random act for all algos
# TODO: remove baselines dependency
# TODO: change target network copying to use deepcopy everywhere
# TODO: move/add parameter freezing to declare network for target nets for all algorithms
# TODO: fix inference hparam
# TODO: fix render hparam

from utils.wrappers import make_envs_general
from utils.plot import plot_reward
from utils.hyperparameters import Config
from utils import create_directory, save_config, update_linear_schedule
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import random
import os
import argparse
import gym
import pybulletgym

gym.logger.set_level(40)


parser = argparse.ArgumentParser(description='RL')
# Meta Info
parser.add_argument('--device', default='cuda',
                    help='device to train on (default: cuda)')
parser.add_argument('--algo', default='dqn',
                    help='algorithm to use: dqn | c51 | a2c | ppo | sac')
parser.add_argument('--env-id', default='BreakoutNoFrameskip-v4',
                    help='environment to train on (default: BreakoutNoFrameskip-v4)')
parser.add_argument('--seed', type=int, default=None, help='random seed. \
                        Note if seed is None then it will be randomly \
                        generated (default: None)')
parser.add_argument('--inference', action='store_true', default=False,
                    help='[NOT WORKING] Inference saved model.')
parser.add_argument('--print-threshold', type=int, default=1000,
                    help='print progress and plot every print-threshold timesteps (default: 1000)')
parser.add_argument('--save-threshold', type=int, default=100000,
                    help='save nn params every save-threshold timesteps (default: 100000)')
parser.add_argument('--render', action='store_true', default=False,
                    help='[NOT WORKING] Render the inference epsiode (default: False')
parser.add_argument('--logdir', default='./results/train/',
                                        help='algorithm to use (default: ./results/train)')
parser.add_argument('--correct-time-limits', action='store_true', default=False,
                    help='Ignore time-limit end of episode when updating (default: False')

# Preprocessing
parser.add_argument('--stack-frames', type=int, default=4,
                    help='[Atari Only] Number of frames to stack (default: 4)')
parser.add_argument('--adaptive-repeat', nargs='+', type=int, default=[4],
                    help='[Atari Only] Possible action repeat values (default: [4])')
parser.add_argument('--state-norm', type=float, default=255.0,
                    help='Normalization constant for states. Set to None if normalization \
                        is handled elsewhere (wrappers) or unneeded (default: 255.0)')
parser.add_argument('--sticky-actions', type=float, default=0.,
                    help='[Atari Only] Sticky action probability. I.e. the probability that \
                        input is ignored and the previous action is repeated \
                        (default: 0.)')

# Learning Control Variables
parser.add_argument('--max-tsteps', type=int, default=2e7,
                    help='Maximimum number of timsteps to train (default: 2e7)')
parser.add_argument('--learn-start', type=int, default=8e4,
                    help='tstep to start updating (default: 80000)')
parser.add_argument('--random-act', type=int, default=1e4,
                    help='[SAC Only] Take uniform random actions until this tstep (default: 10000)')
parser.add_argument('--nenvs', type=int, default=1,
                    help='number of parallel environments executing (default: 1)')
parser.add_argument('--update-freq', type=int, default=4,
                    help='frequency (tsteps) to perform updates (default: 4)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--anneal-lr', action='store_true', default=False,
                    help='anneal lr from start value to 0 throught training')
parser.add_argument('--grad-norm-max', type=float, default=40.0,
                    help='max norm of gradients (default: 40.0)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')

# Optimizer Parameters
parser.add_argument('--rms-alpha', type=float, default=0.99,
                    help='[A2C ONLY] alpha param of rmsprop, used in a2c (default: 0.99)')
parser.add_argument('--optim-eps', type=float, default=1e-4,
                    help='epsilon param of optimizer (default: 1e-4)')

# Replay Memory
parser.add_argument('--replay-size', type=int, default=1e6,
                    help='!!!WATCH YOUR RAM USAGE!!! Size of replay buffer (default: 1000000)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Size of minibatches drawn from replay buffer (default: 32)')
parser.add_argument('--tnet-update', type=int, default=4e4,
                    help='[Doesn\'t affect SAC] Num Steps between target net updates (default: 40000)')
parser.add_argument('--polyak-coef', type=float, default=0.995,
                    help='[SAC ONLY] \theta_targ <- polyak_coef*\theta_targ + (1.-polyak_coef)*\theta\
                         while using polyak averaging in SAC (default: 0.995)')

# Epsilon Variables
parser.add_argument('--eps-start', type=float, default=1.0,
                    help='[\eps-greedy algs only] starting value of epsilon (default: 1.0)')
parser.add_argument('--eps-end', nargs='+', type=float, default=[0.1, 0.01],
                    help='[\eps-greedy algs only] ending value of epsilon for each part of the peicewise function (default: [0.1, 0.01])')
parser.add_argument('--eps-decay', nargs='+', type=float, default=[0.05, 0.5],
                    help='[\eps-greedy algs only] Percent of training at which each eps-end value will be reached via linear decay\n. Choose \
                        a single value > 1.0 to switch to an exponential decay schedule (default: [0.05, 0.5])')

# Nstep
parser.add_argument('--n-steps', type=int, default=1,
                    help='[Exp-Replay Only] Value of N used in N-Step Q-Learning (default: 1)')

# Double DQN
parser.add_argument('--double-dqn', action='store_true', default=False,
                    help='[DQN Only] Use double learning with dqn')

# Priority Replay
parser.add_argument('--priority-replay', action='store_true', default=False,
                    help='[Replay Only] Use prioritized replay with dqn')
parser.add_argument('--priority-alpha', type=float, default=0.6,
                    help='[Replay Only] Alpha value of prioritized replay (default: 0.6)')
parser.add_argument('--priority-beta-start', type=float, default=0.4,
                    help='[Replay Only] starting value of beta in prioritized replay (default: 0.4)')
parser.add_argument('--priority-beta-steps', type=int, default=2e7,
                    help='[Replay Only] steps over which to anneal priority beta to 1 (default: 2e7)')

# Dueling DQN
parser.add_argument('--dueling-dqn', action='store_true', default=False,
                    help='[DQN Only] Use dueling architecture with dqn')

# Noisy Nets
parser.add_argument('--noisy-nets', action='store_true', default=False,
                    help='Use noisy networks for exploration (all algorithms)')
parser.add_argument('--noisy-sigma', type=float, default=0.5,
                    help='Initial sigma value for noisy networks (default: 0.5)')

# Categorical DQN
parser.add_argument('--c51-atoms', type=int, default=51,
                    help='[C51 Only] Number of Atoms in categorical DQN (default: 51)')
parser.add_argument('--c51-vmin', type=float, default=-10.0,
                    help='[C51 Only] Minimum v in C51 (default: -10.0)')
parser.add_argument('--c51-vmax', type=int, default=10.0,
                    help='[C51 Only] Minimum v in C51 (default: 10.0)')

# Quantile DQN

# DRQN

# Recurrent Policy Gradient
parser.add_argument('--recurrent-policy-gradient', action='store_true', default=False,
                    help='[A2C-Style Only] Activate recurrent policy for pg methods')
parser.add_argument('--gru-size', type=int, default=512,
                    help='[A2C-Style Only] number of output units for main gru in pg methods (default: 512)')

# A2C Controls
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='[A2C-Style Only] value loss coefficient for pg methods (default: 0.5)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient for pg methods (default: 0.01)')
parser.add_argument('--entropy-tuning', action='store_true', default=False,
                    help='[SAC ONLY] Automatically tune entropy-coef. Ignores input to --entropy-coef (default: False)')

# GAE Controls
parser.add_argument('--use-gae', action='store_true', default=False,
                    help='[A2C-Style Only] enable generalized advantage estimation for pg methods')
parser.add_argument('--gae-tau', type=float, default=0.95,
                    help='[A2C-Style Only] gae parameter (default: 0.95)')

# PPO Controls
parser.add_argument('--ppo-epoch', type=int, default=3,
                    help='[PPO Only] number of ppo epochs (default: 3)')
parser.add_argument('--ppo-mini-batch', type=int, default=4,
                    help='[PPO Only] number of batches for ppo (default: 4)')
parser.add_argument('--ppo-clip-param', type=float, default=0.1,
                    help='[PPO Only] ppo clip parameter (default: 0.1)')
parser.add_argument('--disable-ppo-clip-value', action='store_true', default=False,
                    help='[PPO Only] DON\'T clip value function in PPO')
parser.add_argument('--disable-ppo-clip-schedule', action='store_true', default=False,
                    help='[PPO Only] DON\'T linearly decay ppo clip by maximum timestep')


def train(config, Agent, valid_arguments, default_arguments, ipynb=False):
    # make/clear directories for logging
    base_dir = os.path.join(config.logdir, config.algo, config.env_id)
    log_dir = os.path.join(base_dir, 'logs/')
    model_dir = os.path.join(base_dir, 'saved_model/')
    tb_dir = os.path.join(base_dir, 'runs/')
    create_directory(base_dir)
    create_directory(log_dir)
    create_directory(model_dir)
    create_directory(tb_dir)

    # Tensorboard writer
    writer = SummaryWriter(log_dir=tb_dir, comment='stuff')

    # save configuration for later reference
    save_config(config, base_dir, valid_arguments)

    # set seeds
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    envs = make_envs_general(config.env_id, config.seed, log_dir,
                             config.nenvs, stack_frames=config.stack_frames,
                             adaptive_repeat=config.adaptive_repeat,
                             sticky_actions=config.sticky_actions, clip_rewards=True)

    agent = Agent(env=envs, config=config, log_dir=base_dir, tb_writer=writer,
        valid_arguments=valid_arguments, default_arguments=default_arguments)

    max_epochs = int(config.max_tsteps / config.nenvs / config.update_freq)

    progress = range(1, max_epochs + 1)
    if not ipynb:
        progress = tqdm.tqdm(range(1, max_epochs + 1), dynamic_ncols=True)
        progress.set_description("Updates %d, Tsteps %d, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
                                 (0, 0, 0, 0, 0.0, 0.0, 0.0))

    start = timer()
    for epoch in progress:

        if config.anneal_lr:
            update_linear_schedule(
                agent.optimizer, epoch-1, max_epochs, config.lr)

        for step in range(config.update_freq):
            # current step for env 0
            current_tstep = (epoch-1)*config.update_freq * \
                config.nenvs + step*config.nenvs

            agent.step(current_tstep, step)

            current_tstep = (epoch) * config.nenvs * config.update_freq
            if current_tstep % config.save_threshold == 0:
                agent.save_w()

            if current_tstep % config.print_threshold == 0 and agent.last_100_rewards:
                update_progress(config, progress, agent, current_tstep, start, log_dir, ipynb)

        if current_tstep > config.learn_start:
            agent.update(current_tstep)

    if(agent.last_100_rewards):
        update_progress(config, progress, agent, config.max_tsteps, start, log_dir, ipynb)

    agent.save_w()
    envs.close()

def update_progress(config, progress, agent, current_tstep, start, log_dir, ipynb):
    end = timer()
    if not ipynb:
        progress.set_description("Upd. %d, Tsteps %d, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
            (
            int(np.max([(current_tstep-config.learn_start)/config.update_freq, 0])),
            current_tstep,
            int(current_tstep * np.mean(config.adaptive_repeat) / (end - start)),
            np.mean(agent.last_100_rewards),
            np.median(agent.last_100_rewards),
            np.min(agent.last_100_rewards),
            np.max(agent.last_100_rewards))
            )
    else:
        plot_reward(log_dir, config.env_id, config.max_tsteps, bin_size=10, smooth=1,
                    time=timedelta(seconds=end-start), save_filename='results.png', ipynb=True)


if __name__ == '__main__':
    args = parser.parse_args()

    # TODO: resume here. Log default parameters
    #   record which algorithms are relevant to each parameter
    #   for each algorithm, throw error when irrelevant parameter
    #   is changed from default

    # get all default arguments
    default_arguments = {}
    for key in vars(args):
        default_arguments[key] = parser.get_default(key)

    # Keep track of valid arguments for each algorithm
    #   as library grows, we can throw errors when invalid
    #   args are changed from default values; this is likely
    #   unintended by the user
    universal_arguments = {
        'device', 'algo', 'env_id', 'seed', 'inference',
        'print_threshold', 'save_threshold', 'render',
        'logdir', 'correct_time_limits', 'state_norm',
        'max_tsteps', 'learn_start', 'random_act', 'nenvs',
        'update_freq', 'lr', 'anneal_lr', 'grad_norm_max',
        'gamma', 'optim_eps', 'noisy_nets', 'noisy_sigma'
    }
    dqn_arguments = {
        'stack_frames', 'adaptive_repeat', 'sticky_actions',
        'replay_size', 'batch_size', 'tnet_update',
        'eps_start', 'eps_end', 'eps_decay', 'n_steps',
        'double_dqn', 'priority_replay', 'priority_alpha',
        'priority_beta_start', 'priority_beta_steps',
        'dueling_dqn'
    }
    c51_arguments = {
        'c51_atoms', 'c51_vmax', 'c51_vmin'
    }
    a2c_arguments = {
        'rms_alpha', 'recurrent_policy_gradient', 'gru_size',
        'entropy_coef', 'value_loss_coef', 'use_gae',
        'gae_tau'
    }
    ppo_arguments = {
        'ppo_epoch', 'ppo_mini_batch', 'ppo_clip_param',
        'disable_ppo_clip_value', 'disable_ppo_clip_schedule'
    }
    sac_arguments = {
        'random_act', 'polyak_coef', 'entropy_coef',
        'entropy_tuning'
    }

    # Import Correct Agent
    if args.algo == 'dqn':
        from agents.DQN import Agent
        valid_arguments = universal_arguments | dqn_arguments
    elif args.algo == 'c51':
        from agents.Categorical_DQN import Agent
        valid_arguments = universal_arguments | dqn_arguments \
            | c51_arguments
    elif args.algo == 'a2c':
        from agents.A2C import Agent
        valid_arguments = universal_arguments | a2c_arguments
    elif args.algo == 'ppo':
        from agents.PPO import Agent
        valid_arguments = universal_arguments | a2c_arguments \
            | ppo_arguments
    elif args.algo == 'sac':
        from agents.SAC import Agent
        valid_arguments = universal_arguments | dqn_arguments
        valid_arguments = valid_arguments - {
            'stack_frames', 'adaptive_repeat', 'sticky_actions',
            'tnet_update', 'eps_start', 'eps_end', 'eps_decay',
            'double_dqn', 'dueling_dqn'
        }
        valid_arguments = valid_arguments | sac_arguments
    else:
        print("INVALID ALGORITHM. ABORT.")
        exit()

    # training params
    config = Config()

    # meta info
    config.device = args.device#
    config.algo = args.algo#
    config.env_id = args.env_id#
    config.seed = args.seed#
    config.inference = args.inference#
    config.print_threshold = int(args.print_threshold)#
    config.save_threshold = int(args.save_threshold)#
    config.render = args.render#
    config.logdir = args.logdir#
    config.correct_time_limits = args.correct_time_limits#

    # preprocessing
    config.stack_frames = int(args.stack_frames)#
    config.adaptive_repeat = args.adaptive_repeat#
    config.state_norm = args.state_norm#
    config.sticky_actions = args.sticky_actions#

    # Learning Control Variables
    config.max_tsteps = int(args.max_tsteps)#
    config.learn_start = int(args.learn_start)#
    config.random_act = int(args.random_act)#
    config.nenvs = int(args.nenvs)#
    config.update_freq = int(args.update_freq)#
    config.lr = args.lr#
    config.anneal_lr = args.anneal_lr#
    config.grad_norm_max = args.grad_norm_max#
    config.gamma = args.gamma#

    # Optimizer params
    config.rms_alpha = args.rms_alpha#
    config.optim_eps = args.optim_eps#

    # memory
    config.replay_size = int(args.replay_size)#
    config.batch_size = int(args.batch_size)#
    config.tnet_update = int(args.tnet_update)#
    config.polyak_coef = float(args.polyak_coef)#

    # epsilon variables
    config.eps_start = args.eps_start#
    config.eps_end = args.eps_end#
    config.eps_decay = args.eps_decay#

    # Multi-step returns
    config.n_steps = int(args.n_steps)#

    # Double DQN
    config.double_dqn = args.double_dqn#

    # Priority Replay
    config.priority_replay = args.priority_replay#
    config.priority_alpha = args.priority_alpha#
    config.priority_beta_start = args.priority_beta_start#
    config.priority_beta_steps = args.priority_beta_steps#

    # Dueling DQN
    config.dueling_dqn = args.dueling_dqn#

    # Noisy Nets
    config.noisy_nets = args.noisy_nets#
    config.noisy_sigma = args.noisy_sigma#

    # Categorical Params
    config.c51_atoms = args.c51_atoms#
    config.c51_vmax = args.c51_vmax#
    config.c51_vmin = args.c51_vmin#

    # Quantile Regression Parameters
    # config.quantiles = 51

    # DRQN Parameters
    # config.drqn_sequence_length = 8

    # Recurrent control
    config.recurrent_policy_gradient = args.recurrent_policy_gradient
    config.gru_size = int(args.gru_size)

    # A2C Controls
    config.entropy_coef = args.entropy_coef
    config.entropy_tuning = args.entropy_tuning #SAC only
    config.value_loss_coef = args.value_loss_coef

    # GAE Controls
    config.use_gae = args.use_gae
    config.gae_tau = args.gae_tau

    # PPO Controls
    config.ppo_epoch = int(args.ppo_epoch)
    config.ppo_mini_batch = int(args.ppo_mini_batch)
    config.ppo_clip_param = args.ppo_clip_param
    config.disable_ppo_clip_value = args.disable_ppo_clip_value
    config.disable_ppo_clip_schedule = args.disable_ppo_clip_schedule

    train(config, Agent, valid_arguments, default_arguments)
