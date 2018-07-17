# DeepRL-Tutorials
The intent of these IPython Notebooks are mostly to help me practice and understand the papers I read; thus, I will opt for readability over efficiency in some cases. First the implementation will be uploaded, followed by markup to explain each portion of code. I'll be assigning credit for any code which is  borrowed in the Acknowledgements section of this README.


Relevant Papers:
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
    
    
Requirements: 

* Python 3.6
* Numpy 
* Gym 
* Pytorch 0.4.0 
* Matplotlib 
* OpenCV 
* Baslines

Acknowledgements: 
* Credit to [@baselines](https://github.com/openai/baselines) for the environment wrappers and inspiration for the prioritized replay code used only in the development code
* Credit to [@higgsfield](https://github.com/higgsfield) for the plotting code, epsilon annealing code, and inspiration for the prioritized replay implementation in the IPython notebook
* Credit to [@Kaixhin](https://github.com/Kaixhin) for factorized Noisy Linear Layer implementation and the projection_distribution function found in Categorical-DQN.ipynb
* Credit to [@ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for A2C, GAE, PPO and visdom plotting code implementation reference
