# TODO:
* Add value of states to tb logging for PPO
* Double check PPO hyperparameters (below)

* Why is Rainbow so slow?
* Fix manual reward logging for all agents
* Fix sig, loss, and action selection freq logging for policy gradient methods
* Finish implementing curious agent
* Update dqn_devel to look more like a2c_devel
* Instead of copying config variables to each agent, copy entire config class and access member variables with self.config.var
* Rename each agent class from "Model" to something more descriptive

# DeepRL-Tutorials
The intent of these IPython Notebooks are mostly to help me practice and understand the papers I read; thus, I will opt for readability over efficiency in some cases. First the implementation will be uploaded, followed by markup to explain each portion of code. I'll be assigning credit for any code which is  borrowed in the Acknowledgements section of this README.


## Relevant Papers:
1. Human Level Control Through Deep Reinforement Learning [[Publication]](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/) [[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/01.DQN.ipynb) 
2. Multi-Step Learning (from Reinforcement Learning: An Introduction, Chapter 7) [[Publication]](http://incompleteideas.net/book/the-book-2nd.html)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/02.NStep_DQN.ipynb) 
3. Deep Reinforcement Learning with Double Q-learning [[Publication]](https://arxiv.org/abs/1509.06461)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/03.Double_DQN.ipynb) 
4. Dueling Network Architectures for Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1511.06581)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/04.Dueling_DQN.ipynb) 
5. Noisy Networks for Exploration [[Publication]](https://arxiv.org/abs/1706.10295)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/05.DQN-NoisyNets.ipynb)
6. Prioritized Experience Replay [[Publication]](https://arxiv.org/abs/1511.05952?context=cs)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/06.DQN_PriorityReplay.ipynb)
7. A Distributional Perspective on Reinforcement Learning [[Publication]](https://arxiv.org/abs/1707.06887)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/07.Categorical-DQN.ipynb)
8. Rainbow: Combining Improvements in Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1710.02298)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/08.Rainbow.ipynb)
9. Distributional Reinforcement Learning with Quantile Regression [[Publication]](https://arxiv.org/abs/1710.10044)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/09.QuantileRegression-DQN.ipynb)
10. Rainbow with Quantile Regression [[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/10.Quantile-Rainbow.ipynb)
11. Deep Recurrent Q-Learning for Partially Observable MDPs [[Publication]](https://arxiv.org/abs/1507.06527)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/11.DRQN.ipynb)
12. Advantage Actor Critic (A2C) [[Publication1]](https://arxiv.org/abs/1602.01783)[[Publication2]](https://blog.openai.com/baselines-acktr-a2c/)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/12.A2C.ipynb)
13. High-Dimensional Continuous Control Using Generalized Advantage Estimation [[Publication]](https://arxiv.org/abs/1506.02438)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/13.GAE.ipynb)
14. Proximal Policy Optimization Algorithms [[Publication]](https://arxiv.org/abs/1707.06347)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/14.PPO.ipynb)

### A2C:
python a2c_devel.py

### Recurrent A2C:
python a2c_devel.py --print-threshold 25 --recurrent-policy --num-steps 20

### PPO:
python a2c_devel.py --algo ppo --print-threshold 10 --save-threshold 100 --lr 2.5e-4 --num-processes 8 --num-steps 128 --enable-gae --disable-ppo-clip-value
    
## Requirements: 

* Python 3.6
* Numpy 
* Gym 
* Pytorch 0.4.0 
* Matplotlib 
* OpenCV 
* Baslines

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
    * Gradient Norm Clipping at 5.0 added

# Acknowledgements: 
* Credit to [@baselines](https://github.com/openai/baselines) for the environment wrappers and inspiration for the prioritized replay code.
* Credit to [@higgsfield](https://github.com/higgsfield) for inspiration for the structure of the notebook, hyperparameter values, and epsilon annealing code.
* Credit to [@Kaixhin](https://github.com/Kaixhin) for factorized Noisy Linear Layer implementation and the projection_distribution function found in Categorical-DQN.ipynb
* Credit to [@ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for A2C, GAE, PPO and visdom plotting code implementation
