[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Collaborative Competition

In this repository, we provide a code implementation of a _M(ulti) A(gent) D(eep) D(eterministic) P(olicy) G(radient)_ approach to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) environment of [unity](unity3d.com). Since the task is of mixed type, i.e. although each agent tries to win, they have to play long enough to reach an average score of 0.5, we coined the task _collaborative competition_. 


![Trained Agent][image1]

The code solving the tennis environment is split up into four parts:
* The `ddpg_agent.py`file is just the DDPG implementation of the [Continuous Control](https://github.com/cmayrhofer/ContinuousControl) problem.
* The `model.py` file is up to some changes in the amount of hidden nodes identical with the one from [Continuous Control](https://github.com/cmayrhofer/ContinuousControl).
* The `maddpg.py` file implements a class for the interplay of the agents.
* The training of the agents, visuallisation of the agents' performance during learning (i.e. learning curve), and the agents playing a test episode, after a successful training, is all done in the jupyter notebook `Tennis.ipynb`. _All the necessary information on how to run the code are provided within this notebook._

Furthermore, the repository contains also the saved weights (and optimizer state) of the two trained actors and critics. They can be found in the file `checkpoint_actors_and_critics.pth`. The pretrained weights can be loaded via [pytorch](pytorch.org) to neural networks with the same architectures as the ones in `model.py`. In the `Report.md` file you can find further informations on the learning algorithm used to solve this environment and how it is implemented in the above listed files.


## Details of the RL Environment

As the above GIF-animation adumbrates, each agent can move its racket into two directions. The goal is to hit the ball and get it to the side of the other agent without letting it drop.

* _`States`_: the state space for each agent is 8 dimensional and consists of the position and velocity of the ball and the racket of the agent;
* _`Actions`_: the agents' action space is (continuous) 2 dimensional and corresponds to moving towards and away from the net and jumping;
* _`Reward`_: an agent obtains a reward of $+0.1$ if he hits the ball over the net and $-0.1$ if he drops the ball or hits the ball out of the bounds.

The environment is to be considered solved if the average score over 100 consecutive episodes is greater equal 0.5. The score for each episode is obtained by taking the maximum of the two scores from the two agents. 

## Getting Started

To run the code provided in this repository, you must have python 3.6 or higher installed. In addition, you will need to have the packages: [numpy](http://www.numpy.org/), [matplotlib](https://matplotlib.org/), [torch](https://pytorch.org/) (including its dependencies) installed. Then follow the next three steps. Afterwards you should be able to run the `Navigation.ipynb` without any dependencies errrors.

1. Follow the steps in [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install the Unity ML-Agents Toolkit.
2. Download the single reacher environment from one of the links below. Select the environment which matches your operating system:
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    
3. Place the file in the folder where you also placed the `maddpg.py` file, the `ddpg_agent.py` file, the `model.py` file, and the `Tennis.ipynb` notebook and unzip (or decompress) the file.

If you are still having issues after following this steps regarding the dependencies then please check out the more throughly configuration [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).
