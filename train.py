
# TODO: add arg to control type of noise in noisy nets
# TODO: efficiency for priority replay functions
# TODO: Add param to select device
# TODO: fix computation graph for recurrent a2c
# TODO: add hparams to tensorboard
# TODO: add video to tensorboard
# TODO: add inference mode

import gym
gym.logger.set_level(40)

import argparse, os, random, tqdm
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import save_config, update_linear_schedule, create_directory
from utils.wrappers import make_env_atari
from utils.hyperparameters import Config
from utils.plot import plot_reward

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

parser = argparse.ArgumentParser(description='RL')
# Meta Info
parser.add_argument('--algo', default='dqn',
					help='algorithm to use: dqn | c51 | a2c | ppo')
parser.add_argument('--env-name', default='BreakoutNoFrameskip-v4',
					help='environment to train on (default: BreakoutNoFrameskip-v4)')
parser.add_argument('--seed', type=int, default=None, help='random seed. \
                        Note if seed is None then it will be randomly \
                        generated (default: None)')
parser.add_argument('--inference', action='store_true', default=False,
					help='Inference saved model')
parser.add_argument('--print-threshold', type=int, default=250,
					help='print progress and plot every print-threshold timesteps (default: 1000)')
parser.add_argument('--save-threshold', type=int, default=2500,
					help='save nn params every save-threshold timesteps (default: 1000)')
parser.add_argument('--render', action='store_true', default=False,
                    help='Render the inference epsiode (default: False')

# Preprocessing
parser.add_argument('--stack-frames', type=int, default=4,
					help='Number of frames to stack (default: 4)')
parser.add_argument('--adaptive-repeat', nargs='+', type=int, default=[4],
                    help='Possible action repeat values (default: [4])')
parser.add_argument('--state-norm', type=float, default=255.0,
					help='Normalization constant for states. Set to None if normalization \
                        is handled elsewhere (wrappers) or unneeded (default: 255.0)')
parser.add_argument('--sticky-actions', type=float, default=0.,
                    help='Sticky action probability. I.e. the probability that \
                        input is ignored and the previous action is repeated \
                        (default: 0.)')

# Learning Control Variables
parser.add_argument('--max-tsteps', type=int, default=2e7,
					help='Maximimum number of timsteps to train (default: 2e7)')
parser.add_argument('--learn-start', type=int, default=8e4,
					help='tstep to start updating for dqn-type methods only (default: 80000)')
parser.add_argument('--nenvs', type=int, default=1,
					help='number of parallel environments executing (default: 1)')
parser.add_argument('--update-freq', type=int, default=4,
					help='frequency (tsteps) to perform updates (default: 4)')
parser.add_argument('--lr', type=float, default=1e-4,
					help='learning rate (default: 1e-4)')
parser.add_argument('--anneal-lr', action='store_true', default=False,
					help='anneal lr from start value to 0 throught training')
parser.add_argument('--max-grad-norm', type=float, default= 40.0,
					help='max norm of gradients (default: 40.0)')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor for rewards (default: 0.99)')

# RMSProp Parameters
parser.add_argument('--rms-alpha', type=float, default=0.99,
					help='alpha param of rmsprop, used in a2c (default: 0.99)')
parser.add_argument('--rms-eps', type=float, default=1e-5,
					help='epsilon param of rmsprop, used in a2c (default: 1e-5)')

# Adam Parameters
parser.add_argument('--adam-eps', type=float, default=1e-4,
					help='epsilon param of adam (default: 1e-4)')

# Replay Memory
parser.add_argument('--replay-size', type=int, default=1e6,
					help='!!!WATCH YOUR RAM USAGE!!! Size of replay buffer (default: 1000000)')
parser.add_argument('--batch-size', type=int, default=32,
					help='Size of minibatches drawn from replay buffer (default: 32)')
parser.add_argument('--tnet-update', type=int, default=4e4,
					help='Num Steps between target net updates (default: 40000)')
    
# Epsilon Variables
parser.add_argument('--eps-start', type=float, default=1.0,
					help='starting value of epsilon (default: 1.0)')
parser.add_argument('--eps-end', nargs='+', type=float, default=[0.1, 0.01],
					help='ending value of epsilon for each part of the peicewise function (default: [0.1, 0.01])')
parser.add_argument('--eps-decay', nargs='+', type=float, default=[0.05, 0.5],
					help='Percent of training at which each eps-end value will be reached via linear decay\n. Choose \
                        a single value > 1.0 to switch to an exponential decay schedule (default: [0.05, 0.5])')

# Nstep
parser.add_argument('--n-steps', type=int, default=1,
					help='Value of N used in N-Step Q-Learning (default: 1)')

# Double DQN
parser.add_argument('--double-dqn', action='store_true', default=False,
					help='Use double learning with dqn')

# Priority Replay
parser.add_argument('--priority-replay', action='store_true', default=False,
					help='Use prioritized replay with dqn')
parser.add_argument('--priority-alpha', type=float, default=0.6,
					help='Alpha value of prioritized replay (default: 0.6)')
parser.add_argument('--priority-beta-start', type=float, default=0.4,
					help='starting value of beta in prioritized replay (default: 0.4)')
parser.add_argument('--priority-beta-steps', type=int, default=2e7,
					help='steps over which to anneal priority beta to 1 (default: 2e7)')

# Dueling DQN
parser.add_argument('--dueling-dqn', action='store_true', default=False,
					help='Use dueling architecture with dqn')

# Noisy Nets
parser.add_argument('--noisy-nets', action='store_true', default=False,
					help='Use noisy networks for exploration (all algorithms)')
parser.add_argument('--noisy-sigma', type=float, default=0.5,
					help='Initial sigma value for noisy networks (default: 0.5)')

# Categorical DQN
parser.add_argument('--c51-atoms', type=int, default=51,
					help='Number of Atoms in categorical DQN (default: 51)')
parser.add_argument('--c51-vmin', type=float, default=-10.0,
					help='Minimum v in C51 (default: -10.0)')
parser.add_argument('--c51-vmax', type=int, default=10.0,
					help='Minimum v in C51 (default: 10.0)')

# Quantile DQN

# DRQN 

# Recurrent Policy Gradient
parser.add_argument('--recurrent-policy-gradient', action='store_true', default=False,
					help='Activate recurrent policy for pg methods')
parser.add_argument('--gru-size', type=int, default=512,
					help='number of output units for main gru in pg methods (default: 512)')

# A2C Controls
parser.add_argument('--value-loss-coef', type=float, default=0.5,
					help='value loss coefficient for pg methods (default: 0.5)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
					help='entropy term coefficient for pg methods (default: 0.01)')

#GAE Controls
parser.add_argument('--enable-gae', action='store_true', default=False,
					help='enable generalized advantage estimation for pg methods')
parser.add_argument('--gae-tau', type=float, default=0.95,
					help='gae parameter (default: 0.95)')

# PPO Controls
parser.add_argument('--ppo-epoch', type=int, default=3,
					help='number of ppo epochs (default: 3)')
parser.add_argument('--ppo-mini-batch', type=int, default=4,
					help='number of batches for ppo (default: 4)')
parser.add_argument('--ppo-clip-param', type=float, default=0.1,
					help='ppo clip parameter (default: 0.1)')
parser.add_argument('--disable-ppo-clip-value', action='store_false', default=True,
					help='DON\'T clip value function in PPO')
parser.add_argument('--disable-ppo-clip-schedule', action='store_false', default=True,
					help='DON\'T linearly decay ppo clip by maximum timestep')

def train(config, Agent, ipynb=False):
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
    writer = SummaryWriter(log_dir=tb_dir, comment='stuff')
    
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

    agent = Agent(env=envs, config=config, log_dir=base_dir, tb_writer=writer)

    max_epochs = int(config.max_tsteps / config.num_envs / config.update_freq)

    progress = range(1, max_epochs + 1)
    if not ipynb:
        progress = tqdm.tqdm(range(1, max_epochs + 1))
        progress.set_description("Updates %d, Tsteps %d, Time %.2f, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
            (0, 0 , 0, 0, 0.0, 0.0, 0.0, 0.0))

    start = timer()
    for epoch in progress:

        if config.use_lr_schedule:
            update_linear_schedule(agent.optimizer, epoch-1, max_epochs, config.lr)

        for step in range(config.update_freq):
            # current step for env 0
            current_tstep = (epoch-1)*config.update_freq*config.num_envs + step*config.num_envs
            
            agent.step(current_tstep, step)

        current_tstep = (epoch) * config.num_envs * config.update_freq 
        agent.update(current_tstep)

        if epoch % config.save_threshold == 0:
                agent.save_w()

        if epoch % config.print_threshold == 0 and agent.last_100_rewards:
            end = timer()
            if not ipynb:
                progress.set_description("Upd. %d, Tsteps %d, Time %s, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
                    (int(np.max([(current_tstep-config.learn_start)/config.update_freq, 0])),
                    current_tstep,
                    str(timedelta(seconds=end-start)).split('.')[0],
                    int(current_tstep*np.mean(config.adaptive_repeat) / (end - start)),
                    np.mean(agent.last_100_rewards),
                    np.median(agent.last_100_rewards),
                    np.min(agent.last_100_rewards),
                    np.max(agent.last_100_rewards))
                )
            else:
                clear_output(True)
                plot_reward(log_dir, config.env_id, config.max_tsteps, bin_size=10, smooth=1, \
                    time=timedelta(seconds=end-start), save_filename='results.png', ipynb=True)
            

    end = timer()
    if(agent.last_100_rewards):
        if not ipynb:
            progress.set_description("Upd. %d, Tsteps %d, Time %s, FPS %d, mean/median R %.1f/%.1f, min/max R %.1f/%.1f" %
                (int(np.max([(config.max_tsteps-config.learn_start)/config.update_freq, 0])),
                config.max_tsteps,
                str(timedelta(seconds=end-start)).split('.')[0],
                int(config.max_tsteps*np.mean(config.adaptive_repeat) / (end - start)),
                np.mean(agent.last_100_rewards),
                np.median(agent.last_100_rewards),
                np.min(agent.last_100_rewards),
                np.max(agent.last_100_rewards))
            )
        else:
            clear_output(True)
            plot_reward(log_dir, config.env_id, config.max_tsteps, bin_size=10, smooth=1,
                time=timedelta(seconds=end-start), save_filename='results.png', ipynb=True)
        
    agent.save_w()
    envs.close()

if __name__=='__main__':
    args = parser.parse_args()

    #Import Correct Agent
    if args.algo == 'dqn':
        from agents.DQN import Agent
    elif args.algo == 'c51':
        from agents.Categorical_DQN import Agent
    elif args.algo == 'a2c':
        from agents.A2C import Agent
    elif args.algo == 'ppo':
        from agents.PPO import Agent
    else:
        print("INVALID ALGORITHM. ABORT.")
        exit()

    #training params
    config = Config()

    # meta info
    config.algo            = args.algo
    config.env_id          = args.env_name
    config.seed            = args.seed
    config.inference       = args.inference
    config.print_threshold = int(args.print_threshold)
    config.save_threshold  = int(args.save_threshold)
    config.render          = args.render

    # preprocessing
    config.stack_frames    = int(args.stack_frames)
    config.adaptive_repeat = args.adaptive_repeat
    config.s_norm          = args.state_norm
    config.sticky_actions  = args.sticky_actions

    # Learning Control Variables
    config.max_tsteps      = int(args.max_tsteps)
    config.learn_start     = int(args.learn_start)
    config.num_envs        = int(args.nenvs)
    config.update_freq     = int(args.update_freq)
    config.lr              = args.lr
    config.use_lr_schedule = args.anneal_lr
    config.grad_norm_max   = args.max_grad_norm
    config.gamma           = args.gamma

    # RMSProp params
    config.rms_alpha = args.rms_alpha
    config.rms_eps   = args.rms_eps

    #adam params
    config.adam_eps = args.adam_eps

    #memory
    config.exp_replay_size        = int(args.replay_size)
    config.batch_size             = int(args.batch_size)
    config.target_net_update_freq = int(args.tnet_update)

    #epsilon variables
    config.epsilon_start = args.eps_start
    config.epsilon_final = args.eps_end
    config.epsilon_decay = args.eps_decay

    # Multi-step returns
    config.N_steps = int(args.n_steps)

    # Double DQN
    config.double_dqn = args.double_dqn

    # Priority Replay
    config.priority_replay      = args.priority_replay
    config.priority_alpha       = args.priority_alpha
    config.priority_beta_start  = args.priority_beta_start
    config.priority_beta_tsteps = args.priority_beta_steps

    # Dueling DQN
    config.dueling_dqn = args.dueling_dqn

    # Noisy Nets
    config.noisy_nets = args.noisy_nets
    config.sigma_init = args.noisy_sigma

    # Categorical Params
    config.c51_atoms = args.c51_atoms
    config.c51_vmax  = args.c51_vmax
    config.c51_vmin  = args.c51_vmin

    # Quantile Regression Parameters
    # config.quantiles = 51

    # DRQN Parameters
    # config.drqn_sequence_length = 8

    #Recurrent control
    config.policy_gradient_recurrent_policy = args.recurrent_policy_gradient
    config.gru_size                         = int(args.gru_size)

    # A2C Controls
    config.entropy_loss_weight = args.entropy_coef
    config.value_loss_weight   = args.value_loss_coef

    # GAE Controls
    config.use_gae = args.enable_gae
    config.gae_tau = args.gae_tau

    # PPO Controls
    config.ppo_epoch       = int(args.ppo_epoch)
    config.ppo_mini_batch  = int(args.ppo_mini_batch)
    config.ppo_clip_param  = args.ppo_clip_param
    config.use_ppo_vf_clip = args.disable_ppo_clip_value
    config.anneal_ppo_clip = args.disable_ppo_clip_schedule

    train(config, Agent)