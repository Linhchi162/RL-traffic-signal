"""Epsilon Greedy Exploration Strategy."""

import numpy as np


class EpsilonGreedy:
    """Epsilon Greedy Exploration Strategy."""

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99):
        """Initialize Epsilon Greedy Exploration Strategy."""
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
    '''
    def choose(self, q_table, state, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())
        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        # print(self.epsilon)
        return action
        '''
    def choose(self, q_table, state, action_space):
        """Choose action based on epsilon greedy strategy."""
        if np.random.rand() < self.epsilon:
            action = tuple(action_space.sample())  # Ensure action is a tuple (phase, green_time)
        else:
            action = max(q_table[state], key=q_table[state].get)

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        return action


    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.initial_epsilon
