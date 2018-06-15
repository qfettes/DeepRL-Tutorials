# DeepRL-Tutorials
The intent of these IPython Notebooks are mostly to help me practice and understand the papers I read; thus, I will opt for readability over efficiency in some cases. First the implementation will be uploaded, followed by markup to explain each portion of code. I'll be assigning credit for any code which is  borrowed in the Acknowledgements section of this README.


Relevant Papers:
1. Human Level Control Through Deep Reinforement Learning [[Publication]](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/) [[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/1.DQN.ipynb) 
2. Multi-Step Learning (from Reinforcement Learning: An Introduction, Chapter 7) [[Publication]](http://incompleteideas.net/book/the-book-2nd.html)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/2.NStep_DQN.ipynb) 
3. Deep Reinforcement Learning with Double Q-learning [[Publication]](https://arxiv.org/abs/1509.06461)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/3.Double_DQN.ipynb) 
4. Dueling Network Architectures for Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1511.06581)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/4.Dueling_DQN.ipynb) 
5. Noisy Networks for Exploration [[Publication]](https://arxiv.org/abs/1706.10295)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/5.DQN-NoisyNets.ipynb)
6. Prioritized Experience Replay [[Publication]](https://arxiv.org/abs/1511.05952?context=cs)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/6.DQN_PriorityReplay.ipynb)
7. A Distributional Perspective on Reinforcement Learning [[Publication]](https://arxiv.org/abs/1707.06887)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/7.Categorical-DQN.ipynb)
8. Rainbow: Combining Improvements in Deep Reinforcement Learning [[Publication]](https://arxiv.org/abs/1710.02298)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/8.Rainbow.ipynb)
9. Distributional Reinforcement Learning with Quantile Regression [[Publication]](https://arxiv.org/abs/1710.10044)[[code]](https://github.com/qfettes/DeepRL-Tutorials/blob/master/9.QuantileRegression-DQN.ipynb)
    
    
Requirements: 

* Python 3.6
* Numpy 
* Gym 
* Pytorch 0.4.0 
* Matplotlib 
* OpenCV 

Acknowledgements: 
* Credit to [@baselines](https://github.com/openai/baselines) for the environment wrappers
* Credit to [@higgsfield](https://github.com/higgsfield) for the plotting code
* Credit to [@Kaixhin](https://github.com/Kaixhin) for factorized Noisy Linear Layer implementation and the projection_distribution function found in Categorical-DQN.ipynb
