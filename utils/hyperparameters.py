import torch
import math


class Config(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #meta infor
        self.algo = None
        self.env_id = None
        self.seed = 0
        self.inference = False
        self.print_threshold = 100
        self.save_threshold = 1000
        self.render = False

        #preprocessing
        self.stack_frames = 4
        self.adaptive_repeat = [4]
        self.s_norm = 255.0
        self.sticky_actions = 0.0

        #Learning control variables
        self.max_tsteps = int(1e7)
        self.learn_start = int(8e4)
        self.num_envs = 1
        self.update_freq = 4
        self.lr = 2.5e-4
        self.use_lr_schedule = False
        self.grad_norm_max = 40.0
        self.gamma = 0.99

        #RMSProp params
        self.rms_alpha = 0.99
        self.rms_eps = 1e-5

        #adam params
        self.adm_eps = 1e-4

        # Replay memory
        self.exp_replay_size = int(1e6)
        self.batch_size = 32
        self.target_net_update_freq = int(1e4)

        #epsilon variables
        self.epsilon_start = 1.0
        self.epsilon_final = [0.1, 0.01]
        self.epsilon_decay = [0.1, 1.0]

        #Multi-step returns
        self.N_steps = 1

        # Double Q Learning

        # Dueling DQN

        #Noisy Nets
        self.noisy_nets = False
        self.sigma_init = 0.5

        #priority replay
        self.priority_replay = False
        self.priority_alpha=0.6
        self.priority_beta_start=0.4
        self.priority_beta_tsteps = 100000

        #Categorical Params
        self.atoms = 51
        self.v_max = 10
        self.v_min = -10

        # Quantile Regression Parameters
        self.quantiles = 51

        #DRQN Parameters
        self.drqn_sequence_length = 8

        # Recurrent Policy Gradient Policy
        self.policy_gradient_recurrent_policy = False
        self.gru_size = 512

        # a2c controls
        self.entropy_loss_weight = 0.01
        self.value_loss_weight = 0.5

        # gae control
        self.use_gae = False
        self.gae_tau = 0.95

        #PPO controls
        self.ppo_epoch = 3
        self.ppo_mini_batch = 32
        self.ppo_clip_param = 0.1
        self.use_ppo_vf_clip = False
        self.anneal_ppo_clip = True