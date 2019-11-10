import torch
import math


class DQNConfig(object):
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

        #adam params
        self.adm_eps = 1e-8
        self.grad_norm_max = 10.0

        #algorithm control
        self.noisy_nets      = False
        self.priority_replay = False
        
        #Multi-step returns
        self.N_STEPS = 1

        #epsilon variables
        self.epsilon_start = 1.0
        self.epsilon_final = [0.01]
        self.epsilon_decay = [0.1]

        #misc agent variables
        self.LR=2.5e-4
        self.GAMMA=0.99

        #memory
        self.TARGET_NET_UPDATE_FREQ = 10000
        self.EXP_REPLAY_SIZE = 1000000
        self.BATCH_SIZE = 32
        self.PRIORITY_ALPHA=0.6
        self.PRIORITY_BETA_START=0.4
        self.PRIORITY_BETA_FRAMES = 100000

        #Noisy Nets
        self.sigma_init=0.5

        #Learning control variables
        self.LEARN_START = 10000
        self.MAX_TSTEPS=100000
        self.num_envs=1
        self.UPDATE_FREQ = 1

        #Categorical Params
        self.ATOMS = 51
        self.V_MAX = 10
        self.V_MIN = -10

        #Quantile Regression Parameters
        self.QUANTILES=51

        #DRQN Parameters
        self.SEQUENCE_LENGTH=8

        #data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000

        # environment specific
        self.s_norm = 255.0
    
    
class PolicyConfig(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #noisy nets
        self.noisy_nets=False
        self.sigma_init=0.5

        #meta infor
        self.algo = None
        self.env_id = None

        #POLICY GRADIENT EXCLUSIVE PARAMETERS
        self.recurrent_policy_grad = False
        self.stack_frames = 4
        self.adaptive_repeat = [4]
        self.gru_size = 512

        #PPO controls
        self.ppo_epoch = 3
        self.num_mini_batch = 32
        self.ppo_clip_param = 0.1
        self.use_ppo_vf_clip = True

        #a2c control
        self.num_agents=8
        self.rollout=128
        self.USE_GAE = True
        self.gae_tau = 0.95

        #misc agent variables
        self.entropy_loss_weight=0.01
        self.value_loss_weight=1.0
        self.grad_norm_max = 0.5
        self.GAMMA=0.99
        self.LR=1e-4

        #Learning control variables
        self.MAX_TSTEPS=100000

        #data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000

        #RMSProp params
        self.rms_alpha = 0.99
        self.rms_eps = 1e-5

        #adam params
        self.adam_eps = 0.01

        #training loop params
        self.seed = 0
        self.inference = False
        self.sticky_actions = 0.0
        self.print_threshold = 100
        self.save_threshold = 1000
        self.use_lr_schedule = True
        self.anneal_ppo_clip = True
        self.render = False

        #compatibility
        self.epsilon_by_frame = None