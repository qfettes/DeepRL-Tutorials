import math

import torch


class Config(object):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # meta infor
        self.algo = None
        self.env_id = None
        self.seed = 0
        self.inference = False
        self.print_threshold = 100
        self.save_threshold = 1000
        self.render = False
        self.logdir = './results/train/'
        self.correct_time_limits = False

        # preprocessing
        self.stack_frames = 4
        self.adaptive_repeat = [4]
        self.state_norm = 1.0
        self.sticky_actions = 0.0

        # Learning control variables
        self.max_tsteps = int(1e7)
        self.learn_start = int(8e4)
        self.random_act = int(1e4)
        self.nenvs = 1
        self.update_freq = 4
        self.lr = 2.5e-4
        self.anneal_lr = False
        self.grad_norm_max = 40.0
        self.gamma = 0.99
        self.body_out = 64

        # Optimizer params
        self.rms_alpha = 0.99
        self.optim_eps = 1e-4

        # Replay memory
        self.replay_size = int(1e6)
        self.batch_size = 32
        self.tnet_update = int(1e4)
        self.polyak_coef = 0.995

        # epsilon variables
        self.eps_start = 1.0
        self.eps_end = [0.1, 0.01]
        self.eps_decay = [0.1, 1.0]

        # Multi-step returns
        self.n_steps = 1

        # Double Q Learning
        self.double_dqn = False

        # priority replay
        self.priority_replay = False
        self.priority_alpha = 0.6
        self.priority_beta_start = 0.4
        self.priority_beta_steps = 100000

        # Dueling DQN
        self.dueling_dqn = False

        # Noisy Nets
        self.noisy_nets = False
        self.noisy_sigma = 0.5

        # Categorical Params
        self.c51_atoms = 51
        self.c51_vmax = 10
        self.c51_vmin = -10

        # Quantile Regression Parameters
        self.quantiles = 51

        # DRQN Parameters
        self.drqn_sequence_length = 8

        # Recurrent Policy Gradient Policy
        self.recurrent_policy_gradient = False
        self.gru_size = 512

        # a2c controls
        self.entropy_coef = 0.01
        self.entropy_tuning = False
        self.value_loss_coef = 0.5

        # gae control
        self.use_gae = False
        self.gae_tau = 0.95

        # PPO controls
        self.ppo_epoch = 3
        self.ppo_mini_batch = 4
        self.ppo_clip_param = 0.1
        self.disable_ppo_clip_value = False
        self.disable_ppo_clip_schedule = False
