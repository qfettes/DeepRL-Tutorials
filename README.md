# TODO:
* Add value of states to tb logging for PPO
* Why is Rainbow so slow?
* Finish implementing curious agent
* Rename each agent class from "Agent" to something more descriptive
* Remove baselines dependency
* Fix scaling in policy gradient methods. Values need to be scaled 0-1
* Remove the if(self.first_action) garbage
* Noisy Nets for Exploration isn't actually implemented for all(?) policy gradient algorithms
* Takes uniform random actions at start of every training session in SAC: My implementation needs to add this

# DeepRL-Tutorials
The intent of these IPython Notebooks are mostly to help me practice and understand the papers I read; thus, I will opt for readability over efficiency in some cases. First the implementation will be uploaded, followed by markup to explain each portion of code. I'll be assigning credit for any code which is  borrowed in the Acknowledgements section of this README.


## Relevant Papers:
1. Human Level Control Through Deep Reinforement Learning [[Publication]](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/) [[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/01.DQN.ipynb) 
2. Multi-Step Learning (from Reinforcement Learning: An Introduction, Chapter 7) [[Publication]](http://incompleteideas.net/book/the-book-2nd.html)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/02.NStep_DQN.ipynb) 
3. Deep Reinforcement Learning with Double Q-learning [[Publication]](https://arxiv.org/abs/1509.06461)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/03.Double_DQN.ipynb) 
4. Prioritized Experience Replay [[Publication]](https://arxiv.org/abs/1511.05952?context=cs)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/06.DQN_PriorityReplay.ipynb)
5. Dueling Network Architectures for Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1511.06581)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/04.Dueling_DQN.ipynb) 
6. Noisy Networks for Exploration [[Publication]](https://arxiv.org/abs/1706.10295)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/05.DQN-NoisyNets.ipynb)
7. A Distributional Perspective on Reinforcement Learning [[Publication]](https://arxiv.org/abs/1707.06887)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/07.Categorical-DQN.ipynb)
8. Rainbow: Combining Improvements in Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1710.02298)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/08.Rainbow.ipynb)
9. Distributional Reinforcement Learning with Quantile Regression [[Publication]](https://arxiv.org/abs/1710.10044)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/09.QuantileRegression-DQN.ipynb)
10. Rainbow with Quantile Regression [[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/10.Quantile-Rainbow.ipynb)
11. Deep Recurrent Q-Learning for Partially Observable MDPs [[Publication]](https://arxiv.org/abs/1507.06527)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/11.DRQN.ipynb)
12. Advantage Actor Critic (A2C) [[Publication1]](https://arxiv.org/abs/1602.01783)[[Publication2]](https://blog.openai.com/baselines-acktr-a2c/)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/12.A2C.ipynb)
13. High-Dimensional Continuous Control Using Generalized Advantage Estimation [[Publication]](https://arxiv.org/abs/1506.02438)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/13.GAE.ipynb)
14. Proximal Policy Optimization Algorithms [[Publication]](https://arxiv.org/abs/1707.06347)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/14.PPO.ipynb)

## Training without Ipython Notebook
Below is a list of commands to train all algorithms without the use of Ipython notebooks.

### DQN (Quick Verification):
```
python train.py --env-name PongNoFrameskip-v4 --stack-frames 1 --eps-end 0.01 --eps-decay 30000 --tnet-update 1000 --replay-size 100000 --learn-start 10000 --max-tsteps 1000000 --update-freq 1 --adam-eps 1.0e-8
```

### DQN
```
python train.py
```

### N-Step DQN
```
python train.py --n-steps 3
```

### Double DQN
```
python train.py --double-dqn
```

### Prioritized Replay
```
python train.py --double-dqn --priority-replay
```

### Dueling DQN
```
python train.py --double-dqn --priority-replay --dueling-dqn --max-grad-norm 10.0
```

### Noisy Networks
```
python train.py --double-dqn --priority-replay --dueling-dqn --max-grad-norm 10.0 --noisy-nets
```

### A2C:
```
python train.py --algo a2c --print-threshold 100 --max-tsteps 10000000 --learn-start 0 --nenvs 16 --update-freq 5 --lr 0.0007 --anneal-lr --max-grad-norm 0.5 
```

### Recurrent A2C:
```
python train.py --algo a2c --print-threshold 25 --save-threshold 250 --max-tsteps 10000000 --learn-start 0 --nenvs 16 --update-freq 20 --lr 0.0007 --anneal-lr --max-grad-norm 0.5 --recurrent-policy-gradient
```

### PPO:
```
python train.py --algo ppo --print-threshold 10 --save-threshold 100 --max-tsteps 10000000 --learn-start 0 --nenvs 8 --update-freq 128 --lr 2.5e-4 --anneal-lr --max-grad-norm 0.5 --adam-eps 1.0e-5 --value-loss-weight 1.0 --enable-gae --disable-ppo-clip-value
```

### Continuous A2C (Not Working):
```
python train.py --env-name Reacher-v2 --algo a2c --print-threshold 25 --save-threshold 250 --max-tsteps 10000000 --learn-start 0 --nenvs 16 --update-freq 200 --lr 0.0007 --anneal-lr --max-grad-norm 0.5 --body-out 200 --gru-size 128 --recurrent-policy-gradient
```

### Continuous PPO:
```
python train.py --env-name Reacher-v2 --algo ppo --print-threshold 1 --save-threshold 10 --max-tsteps 10000000 --learn-start 0 --nenvs 1 --update-freq 2048 --lr 3.0e-4 --anneal-lr --max-grad-norm 0.5 --adam-eps 1.0e-5 --value-loss-coef 0.5 --entropy-coef 0.0 --enable-gae --disable-ppo-clip-value --ppo-epoch 10 --ppo-mini-batch 32 --ppo-clip-param 0.2 --body-out 200 --gru-size 128
```

### Soft Actor-Critic
```
python train.py --algo sac --env-name HalfCheetah-v2 --gamma 0.99 --polyak-coef 0.995 --lr 0.003 --entropy-coef 0.2 --batch-size 100 --max-tsteps 1000000 --learn-start 1000 --random-act 10000 --replay-size 1000000 --update-freq 50 --state-norm 1.0 --adam-eps 1e-8 --correct-time-limits
```
    
## Requirements: 

* Python 3.6
* Numpy 
* matplotlib
* Gym 
* Pytorch>=1.3.0
* OpenCV 
* stable-baselines
* tensorboard

# Known Reproduction Differences
* DQN
    * Learning Rate is 1e-4
        * Original: 2.5e-4
    * The Adam Optimizer is used
        * Original: RMSProp
    * Exploration uses piecewise epsilon schedule, which anneals all the way to 0.01 by the end of training
        * Original: Linear epsilon schedule annealed to 0.1 after 10% of training
    * Learning starts at 80000 timesteps
        * Original: 50000 timesteps
    * Target network updates after every 40000 updates to the online network
        * Original: 10000 updates
    * Number of training frames is 2e7
        * Original: 5e7
    * Huber Loss is used as the loss function
        * Original: Mean Squared Error
    * There is no TD Error Clipping
    * There are no seperate training and evaluation phases
        * Original: Evaluates with a lower epsilon and no learning for 125000 timesteps, every 500000 timesteps
* N-step DQN
    * While this does (seemingly) match how it was described in the RAINBOW paper, it is mathematically incorrect
    * True N-step learning would require off policy correction, this does not
    * All DQN differences apply here, too.
* Double DQN
    * All DQN Differences apply here
    * TNet update is 40000
        * Original: 30000
    * There is no evaluation stage
        * Original: Evaluation Epsilon is 0.001
    * No shared bias in Output layer
        * Original: shared bias between every node in output layer
* Prioritized Replay
    * All Double DQN Differences Apply Here
* Dueling DQN
    * All Prioritized Replay differences apply here
    * Higher Learning rate?
        * Original: Unknown, but the paper notes they use a lower learning rate than Double DQN
    * No gradient rescaling for convolutional layers
        * Original: Rescales gradients entering final convolutional layer by 1/sqrt(2)
* Noisy Networks for Exploration
    * All Prioritized Replay differences apply here

# Acknowledgements: 
* Credit to [@higgsfield](https://github.com/higgsfield) for inspiration for the structure of the entire notebook. Check out this tutorial too!
* Credit to [@baselines](https://github.com/openai/baselines) for the original environment wrappers, experience replay code, prioritized replay code,  and Segment Tree implenation. Modifications were made.
* Credit to [@apaszke](https://github.com/apaszke) for the code to mask 'None' next states used in all DQN variants; this was originally viewed in the pytorch tutorial for DQNs.
* Credit to [@Kaixhin](https://github.com/Kaixhin) for factorized Noisy Linear Layer implementation and the projection_distribution function found in Categorical-DQN implementations. Modifications were made.
* Credit to [@ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for A2C, GAE, PPO and the plotting code implementation used in the ipynb notebooks; my 'implentations' here would be better named as adaptions of the original code.
