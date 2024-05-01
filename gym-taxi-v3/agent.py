import random
import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, alpha, nA=6, gamma=1.0):
        """Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state, i_episode):
        """Given the state, select an action.

        Params
        ======
        - state: the current state of the environment
        - i_episode: ith episode number

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        eps = 1.0 / i_episode
        if random.random() > eps:  # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self.nA))

    def step(self, i_episode, state, action, reward, next_state):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.update_Q_expsarsa(
            i_episode, state, action, reward, next_state
        )

    def update_Q_expsarsa(self, i_episode, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        eps = 1.0 / i_episode
        current = self.Q[state][
            action
        ]  # estimate in Q-table (for current state, action pair)
        policy_s = (
            np.ones(self.nA) * eps / self.nA
        )  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = (
            1 - eps + (eps / self.nA)
        )  # greedy action
        Qsa_next = np.dot(
            self.Q[next_state], policy_s
        )  # get value of state at next time step
        target = reward + (self.gamma * Qsa_next)  # construct target
        new_value = current + (self.alpha * (target - current))  # get updated value
        return new_value
