import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, eps_min=.0001, alpha=.99, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - eps: fixed value if not-None, otherwise, decay
        - eps_min: float for min value of epsilon decay
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps_min = eps_min
        self.n_episodes = 1
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        eps = self._get_epsilon()
        Q = self.Q
        
        if np.random.random() > eps: # select greedy action with probability epsilon
            return np.argmax(Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(np.arange(self.nA))
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        epsilon = self._get_epsilon()
        nA = self.nA
        alpha = self.alpha - self.n_episodes * (self.alpha - 0.1)/ 20000.0 
        gamma = self.gamma
        Q = self.Q
        
        if not done:
            action_probs = np.ones(nA) * epsilon / nA
            action_probs[np.argmax(Q[state])] += (1 - epsilon)
            next_state_expected_value = np.dot(Q[next_state], action_probs)
            Q[state][action] = Q[state][action] + alpha * (reward + gamma * next_state_expected_value - Q[state][action])
        else:
            Q[state][action] = Q[state][action] + alpha * (reward - Q[state][action])
            self.n_episodes += 1
        
    def _get_epsilon(self):
        epsilon = 1.0 / self.n_episodes
        if self.eps_min:
            epsilon = max(self.eps_min, epsilon)
        return epsilon