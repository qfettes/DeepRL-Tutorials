#NOTE: A2C works well w/ recurrency if num-steps is 20
# TODO: incorporate recent changes to dqn_devel
#   add parsing of args to main and tqdm
# TODO: add dummyvecenv to here when single env used
# TODO: merge to use same config object as dqn_devel
# TODO roll innermost stuff into a step function 
#   of the agent for more modularity
# TODO: Frames need scaled now

import gym
gym.logger.set_level(40)

import argparse, os
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from utils import save_config, update_linear_schedule, create_directory
from utils.wrappers import make_env_atari
from utils.hyperparameters import PolicyConfig


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
parser.add_argument('--algo', default='a2c',
					help='algorithm to use: a2c | ppo')
parser.add_argument('--print-threshold', type=int, default=100,
					help='print progress and plot every print-threshold timesteps (default: 100)')
parser.add_argument('--save-threshold', type=int, default=1000,
					help='save nn params every save-threshold timesteps (default: 1000)')
parser.add_argument('--lr', type=float, default=7e-4,
					help='learning rate (default: 7e-4)')
parser.add_argument('--gamma', type=float, default=0.99,
					help='discount factor for rewards (default: 0.99)')
parser.add_argument('--max-tsteps', type=int, default=1e7,
					help='number of timesteps to train (default: 1e7)')
parser.add_argument('--num-processes', type=int, default=16,
					help='how many training CPU processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5,
					help='number of forward steps in A2C (default: 5)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
					help='value loss coefficient (default: 0.5)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
					help='entropy term coefficient (default: 0.01)')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
					help='max norm of gradients (default: 0.5)')
parser.add_argument('--rms-alpha', type=float, default=0.99,
					help='alpha param of rmsprop, used in a2c (default: 0.99)')
parser.add_argument('--rms-eps', type=float, default=1e-5,
					help='epsilon param of rmsprop, used in a2c (default: 1e-5)')
parser.add_argument('--adam-eps', type=float, default=1e-5,
					help='epsilon param of adam optimizer, used in ppo (default: 1e-5)')
parser.add_argument('--enable-gae', action='store_true', default=False,
					help='enable generalized advantage estimation')
parser.add_argument('--tau', type=float, default=0.95,
					help='gae parameter (default: 0.95)')
parser.add_argument('--ppo-epoch', type=int, default=4,
					help='number of ppo epochs (default: 4)')
parser.add_argument('--num-mini-batch', type=int, default=4,
					help='number of batches for ppo (default: 4)')
parser.add_argument('--clip-param', type=float, default=0.1,
					help='ppo clip parameter (default: 0.1)')
parser.add_argument('--disable-ppo-clip-value', action='store_false', default=True,
					help='DON\'T clip value function in PPO')
parser.add_argument('--recurrent-policy', action='store_true', default=False,
					help='Activate recurrent policy')
parser.add_argument('--gru-size', type=int, default=512,
					help='number of output units for main gru (default: 512)')
parser.add_argument('--noisy-nets', action='store_true', default=False,
					help='Use Noisy Networks for Exploration')
parser.add_argument('--sigma-init', type=float, default=0.5,
                    help='Initial Noisy network sigma mag (default: 0.5)')
parser.add_argument('--disable-lr-schedule', action='store_false', default=True,
					help='DON\'T linearly decay lr by maximum timestep')
parser.add_argument('--disable-ppo-clip-schedule', action='store_false', default=True,
					help='DON\'T linearly decay ppo clip by maximum timestep')
parser.add_argument('--inference', action='store_true', default=False,
					help='Inference saved model')
parser.add_argument('--render', action='store_true', default=False,
                    help='Render the inference epsiode (default: False')
args = parser.parse_args()

#Import Correct Model
if args.algo == 'a2c':
    from agents.A2C import Model
elif args.algo == 'ppo':
    from agents.PPO import Model
else:
    print("INVALID ALGORITHM. ABORT.")
    exit()

config = PolicyConfig()
config.algo = args.algo
config.env_id = args.env_name

#noisy nets
config.noisy_nets = args.noisy_nets
config.sigma_init = args.sigma_init

#preprocessing
config.stack_frames = args.stack_frames
config.adaptive_repeat = args.adaptive_repeat #adaptive repeat

#Recurrent control
config.recurrent_policy_grad = args.recurrent_policy
config.gru_size = args.gru_size

if config.recurrent_policy_grad:
    model_architecture = 'recurrent/'
else:
    model_architecture = 'feedforward/'

#ppo control
config.ppo_epoch = args.ppo_epoch
config.num_mini_batch = args.num_mini_batch
config.ppo_clip_param = args.clip_param
config.use_ppo_vf_clip = args.disable_ppo_clip_value

#a2c control
config.num_agents=args.num_processes
config.rollout=args.num_steps
config.USE_GAE = args.enable_gae
config.gae_tau = args.tau

#RMSProp params
config.rms_alpha = args.rms_alpha
config.rms_eps = args.rms_eps

#adam params
config.adam_eps = args.adam_eps

#misc agent variables
config.GAMMA=args.gamma
config.LR=args.lr
config.entropy_loss_weight=args.entropy_coef
config.value_loss_weight=args.value_loss_coef
config.grad_norm_max = args.max_grad_norm

#training loop params
config.seed = args.seed
config.inference = args.inference
config.sticky_actions = args.sticky_actions
config.print_threshold = args.print_threshold
config.save_threshold = args.save_threshold
config.use_lr_schedule = args.disable_lr_schedule
config.anneal_ppo_clip = args.disable_ppo_clip_schedule
config.render = args.render

config.MAX_TSTEPS = args.max_tsteps

max_epochs = int(args.max_tsteps / config.num_agents / config.rollout)

def train(config):
    #make/clear directories for logging
    base_dir = os.path.join('./results/', config.algo, model_architecture, config.env_id)
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
    seed = np.random.randint(0, int(1e6)) if config.seed is None else config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    envs = [make_env_atari(config.env_id, seed, i, log_dir, stack_frames=config.stack_frames, adaptive_repeat=config.adaptive_repeat, sticky_actions=config.sticky_actions, clip_rewards=True) for i in range(config.num_agents)]
    envs = SubprocVecEnv(envs)

    model = Model(static_policy=config.inference, env=envs, config=config, log_dir=base_dir, tb_writer=writer)

    obs = envs.reset()
    
    obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)

    model.config.rollouts.observations[0].copy_(obs)
    
    episode_rewards = np.zeros(config.num_agents, dtype=np.float)
    final_rewards = np.zeros(config.num_agents, dtype=np.float)
    last_100_rewards = deque(maxlen=100)

    start = timer()
    
    for epoch in range(1, max_epochs+1):
        if config.use_lr_schedule:
            update_linear_schedule(model.optimizer, epoch-1, max_epochs, config.LR)

        for step in range(config.rollout):
            with torch.no_grad():
                values, actions, action_log_prob, states = model.get_action(
                                                            model.config.rollouts.observations[step],
                                                            model.config.rollouts.states[step],
                                                            model.config.rollouts.masks[step])
            
            cpu_actions = actions.view(-1).cpu().numpy()
    
            obs, reward, done, info = envs.step(cpu_actions)

            obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)

            #agent rewards
            episode_rewards += reward
            masks = 1. - done.astype(np.float32)
            final_rewards *= masks
            final_rewards += (1. - masks) * episode_rewards
            episode_rewards *= masks

            for index, inf in enumerate(info):
                current_tstep = (epoch-1)*config.rollout*config.num_agents+step*config.num_agents+index
                if 'episode' in inf.keys():
                    last_100_rewards.append(inf['episode']['r'])
                    writer.add_scalar('Performance/Environment Reward', inf['episode']['r'], current_tstep)
                    writer.add_scalar('Performance/Episode Length', inf['episode']['l'], current_tstep)

                if done[index]:
                    #write reward on completion
                    writer.add_scalar('Performance/Agent Reward', final_rewards[index], current_tstep)

            rewards = torch.from_numpy(reward.astype(np.float32)).view(-1, 1).to(config.device)
            masks = torch.from_numpy(masks).to(config.device).view(-1, 1)

            obs *= masks.view(-1, 1, 1, 1)

            model.config.rollouts.insert(obs, states, actions.view(-1, 1), action_log_prob, values, rewards, masks)
            
        with torch.no_grad():
            next_value = model.get_values(model.config.rollouts.observations[-1],
                                model.config.rollouts.states[-1],
                                model.config.rollouts.masks[-1])
            
        value_loss, action_loss, dist_entropy, dynamics_loss = model.update(model.config.rollouts, next_value, epoch*config.rollout*config.num_agents)
        
        model.config.rollouts.after_update()

        if epoch % config.save_threshold == 0:
            model.save_w()

        if epoch % config.print_threshold == 0 and len(last_100_rewards) > 0:
            #print
            end = timer()
            total_num_steps = (epoch) * config.num_agents * config.rollout
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, val loss {:.5f}, pol loss {:.5f}, dyn loss {:.5f}".
                format(epoch, total_num_steps,
                       int(total_num_steps*np.mean(config.adaptive_repeat) / (end - start)),
                       np.mean(last_100_rewards),
                       np.median(last_100_rewards),
                       np.min(last_100_rewards),
                       np.max(last_100_rewards), dist_entropy,
                       value_loss, action_loss, dynamics_loss))

    model.save_w()
    envs.close()

def test(config):
    base_dir = os.path.join('./results/', config.algo, model_architecture, config.env_id)
    log_dir = os.path.join(base_dir, 'logs/')
    model_dir = os.path.join(base_dir, 'saved_model/')

    seed = np.random.randint(0, int(1e6))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    env = [make_env_atari(config.env_id, seed, config.num_agents, log_dir, stack_frames=config.stack_frames, adaptive_repeat=config.adaptive_repeat, sticky_actions=config.sticky_actions, clip_rewards=False)]
    env = SubprocVecEnv(env)

    model = Model(env=env, config=config, log_dir=base_dir, static_policy=config.inference)
    model.load_w()

    obs = env.reset()
    
    if config.render:
        env.render()
    
    obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)
    state = model.config.rollouts.states[0, 0].view(1, -1)
    mask = model.config.rollouts.masks[0, 0].view(1, -1)
    
    episode_rewards = np.zeros(1, dtype=np.float)
    final_rewards = np.zeros(1, dtype=np.float)

    start=timer()

    done = False
    tstep=0
    while not done:
        tstep+=1
        with torch.no_grad():
                value, action, action_log_prob, state = model.get_action(obs, state, mask)
            
        cpu_action = action.view(-1).cpu().numpy()
        obs, reward, done, info = env.step(cpu_action)

        if config.render:
            env.render()

        obs = torch.from_numpy(obs.astype(np.float32)).to(config.device)

        episode_rewards += reward
        mask = 1. - done.astype(np.float32)
        final_rewards += (1. - mask) * episode_rewards

        mask = torch.from_numpy(mask).to(config.device).view(-1, 1)

        
    #print
    end = timer()
    total_num_steps = tstep
    print("Num timesteps {}, FPS {}, Reward {:.1f}".
        format(total_num_steps,
                int(total_num_steps*np.mean(config.adaptive_repeat) / (end - start)),
                np.mean(final_rewards)))
    env.close()
            
    
if __name__=='__main__':
    if not config.inference:
        train(config)
    else:
        test(config)
