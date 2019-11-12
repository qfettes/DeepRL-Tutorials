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
        self.sticky_actions = 0.0
        self.print_threshold = 100
        self.save_threshold = 1000
        self.use_lr_schedule = False
        self.render = False

        #preprocessing
        self.stack_frames = 4
        self.adaptive_repeat = [4]

        #Noisy Nets
        self.noisy_nets = False
        self.sigma_init = 0.5

        # Recurrent Policy Gradient Policy
        self.policy_gradient_recurrent_policy = False
        self.gru_size = 512

        #PPO controls
        self.ppo_epoch = 3
        self.num_mini_batch = 32
        self.ppo_clip_param = 0.1
        self.use_ppo_vf_clip = True
        self.anneal_ppo_clip = True

        #Learning control variables
        self.learn_start = 10000
        self.max_tsteps = 100000
        self.num_envs = 1
        self.update_freq = 1

        # gae control
        self.use_gae = True
        self.gae_tau = 0.95

        # a2c controls
        self.entropy_loss_weight = 0.01
        self.value_loss_weight = 1.0
        self.grad_norm_max = 0.5

        #misc agent variables
        self.lr = 2.5e-4
        self.grad_norm_max = 10.0
        self.gamma = 0.99

        #RMSProp params
        self.rms_alpha = 0.99
        self.rms_eps = 1e-5

        #adam params
        self.adm_eps = 1e-8
        
        #Multi-step returns
        self.N_STEPS = 1

        #epsilon variables
        self.epsilon_start = 1.0
        self.epsilon_final = [0.01]
        self.epsilon_decay = [0.1]

        #memory
        self.TARGET_NET_UPDATE_FREQ = 10000
        self.EXP_REPLAY_SIZE = 1000000
        self.BATCH_SIZE = 32

        #priority replay
        self.priority_replay = False
        self.priority_alpha=0.6
        self.priority_beta_start=0.4
        self.priority_beta_tsteps = 100000
        

        #Categorical Params
        self.ATOMS = 51
        self.V_MAX = 10
        self.V_MIN = -10

        #Quantile Regression Parameters
        self.QUANTILES=51

        #DRQN Parameters
        self.SEQUENCE_LENGTH=8

        # environment specific
        self.s_norm = 255.0