[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

## Project Details 

For this project, we worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

#### Rewards
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each time step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

#### State and Action Space
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, two separate versions of the Unity environment were provided:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

I chose the second version with 20 identical agents.

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

#### Solving the Environment 

To solve the environment with 20 agents, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Getting Started
#### Setting Up Your Python Environment
This code was developed using Python 3.6.15

To manage the Python environment, I used [pyenv](https://github.com/pyenv/pyenv#installation) coupled with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installation)
1. Install Python 3.6.15
    - `pyenv install 3.6.15`
2. Create a virtual environment called `p1_navigation` that uses Python 3.6.15:
    - `pyenv virtualenv 3.6.15 p1_navigation`
    - Note: you can create the environment with a different name, just replace `p1_navigation` in the previous command AND in the `.python-version` file in the root directory
3. Install the required packages:
    - `pip install -r requirements.txt`

#### Setup the Unity Environment
The Reacher20.app of Mac OSX is already in the root directory of this project.  

1. If you're using a different OS, download the
environment from one of the links below:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

## Instructions
The three main files are:
1. `ddpg_agent.py` - this file implements the Agent with an Actor and Critic and has the bulk of the hyperparameters to set
2. `model.py` - this file implements the neural networks for the Actor and Critic
3. `Continuous_Control.ipynb` - the Jupyter Notebook that runs it all

Run `Continuous_Control.ipynb` to train your own agent!

## Attributions
This code was based off starter code provided in the DDPG Pendulum example in the [Udacity deep-reinforcement-learning repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
