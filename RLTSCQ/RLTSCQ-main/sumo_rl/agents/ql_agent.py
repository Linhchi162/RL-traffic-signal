"""Q-learning Agent — compatible with Discrete(max_green) action space."""

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class QLAgent:
    """Tabular Q-learning agent for a Discrete action space.

    Supports environments where action_space = gym.spaces.Discrete(n),
    e.g. choosing a green-phase duration in [min_green, max_green].
    """

    def __init__(
        self,
        starting_state,
        state_space,
        action_space,
        alpha: float = 0.5,
        gamma: float = 0.95,
        exploration_strategy=EpsilonGreedy(),
    ):
        """Initialize Q-learning agent.

        Args:
            starting_state: Initial observation (hashable).
            state_space: Observation space (for reference).
            action_space: gym.spaces.Discrete instance.
            alpha: Learning rate.
            gamma: Discount factor.
            exploration_strategy: Exploration strategy (e.g. EpsilonGreedy).
        """
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.acc_reward = 0.0

        self.q_table: dict = {
            self.state: [0.0 for _ in range(action_space.n)]
        }

    def _ensure_state(self, state):
        """Insert state into Q-table if not present."""
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.action_space.n)]

    def act(self) -> int:
        """Choose action via exploration strategy."""
        self._ensure_state(self.state)
        self.action = self.exploration.choose(
            self.q_table, self.state, self.action_space
        )
        return self.action

    def learn(self, next_state, reward: float, done: bool = False):
        """Q-learning update rule."""
        self._ensure_state(self.state)
        self._ensure_state(next_state)

        s, s1, a = self.state, next_state, self.action
        best_next = max(self.q_table[s1])
        self.q_table[s][a] += self.alpha * (
            reward + self.gamma * best_next * (1 - done) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
