[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Collaborative Competition

In this repository, we provide a code implementation of a _M(ulti) A(gent) D(eep) D(eterministic) P(olicy) G(radient)_ approach to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) environment of [unity](unity3d.com). Since the task is of mixed type, i.e. although each agent tries to win they have to play long enough to reach an average score of 0.5, we coined the task _collaborative competition_. 


![Trained Agent][image1]

The code solving the tennis environment is split up into four parts:
* The `ddpg_agent.py`file is just the DDPG implementation of the [Continuous Control](https://github.com/cmayrhofer/ContinuousControl) problem.
* The `model.py` file is up to some changes in the amount of hidden nodes identical with the one from [Continuous Control](https://github.com/cmayrhofer/ContinuousControl).
* The `maddpg.py` file implements a class for the interplay of the agents.
* The training of the agent, visuallisation of the agents performance during learning (i.e. learning curve), and the agent playing a test episode, after a successful training, is all done in the jupyter notebook `Tennis.ipynb`. _All the necessary information on how to run the code are provided within this notebook._

Furthermore, the repository contains also the saved weights (and optimizer state) of the two trained actors and critics. They can be found in the file `checkpoint_actors_and_critics.pth`. The pretrained weights can be loaded via [pytorch](pytorch.org) to neural networks with the same architectures as the ones in `model.py`. In the `Report.md` file you can find further informations on the learning algorithm used to solve this environment and how it is implemented in the above listed files.


## Details of the RL Environment
