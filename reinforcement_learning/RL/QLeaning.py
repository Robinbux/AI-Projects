import numpy as np

from Environment import Environment, Position, Action


class QLearning:

    def __init__(self, epsilon: float = 0.05, alpha: float = 0.9, gamma: float = 0.95):
        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
        self._env = Environment()
        self._possible_states = self._env.get_possible_start_states()
        # Assign indexes to states, to retrieve from Q-matrix later
        self._state_indexes = dict(zip(self._possible_states, list(range(len(self._possible_states)))))

        # Q-matrix
        self._Q = np.zeros((len(self._possible_states), 4))
        #self._Q = np.random.random((len(self._possible_states), 4))

    def pick_action(self, state: Position) -> Action:
        """Pick action based on e-Greedy"""
        if np.random.uniform(0, 1) < self._epsilon:
            return Action.random()
        # If state is terminal state, action is irrelevant
        if self._env.is_terminal_state(state):
            return Action.UP
        action_values = self._Q[self._state_indexes[state], :]
        return Action(np.random.choice(np.flatnonzero(action_values == action_values.max())))
        #return Action(np.argmax(self._Q[self._state_indexes[state], :]))

    def update(self, state: Position, action: Action, next_state: Position, reward: float) -> None:
        state_idx = self._state_indexes[state]
        action_idx = action.value
        current_state_val = self._Q[state_idx, action_idx]
        if self._env.is_terminal_state(next_state):
            next_state_action_val = 0
        else:
            # Pick highest value from all possible actions for state s'
            temp = self._Q[self._state_indexes[state], :]
            next_state_action_val = np.max(self._Q[self._state_indexes[next_state], :])

        self._Q[state_idx, action_idx] = current_state_val + self._alpha * (
                    reward + self._gamma * next_state_action_val - current_state_val)

    def get_best_actions(self) -> dict[Position, Action]:
        best_actions_for_state = {}
        for state, idx in self._state_indexes.items():
            best_actions_for_state[state] = Action(np.argmax(self._Q[idx, :]))
        return best_actions_for_state
