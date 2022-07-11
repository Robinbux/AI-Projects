from Environment import Environment, Position
from Policies import EquiprobablePolicy


class MonteCarloPolicyEvaluation:

    def __init__(self, nbr_samples: int):
        self._nbr_samples = nbr_samples
        self._env = Environment()

    def evaluate(self, policy: EquiprobablePolicy) -> list[list[float]]:
        value_map: list[list[float]] = [[0]*9 for _ in range(9)]
        start_positions = self._env.get_possible_start_states()
        for idx, state in enumerate(start_positions):
            print(f"Evaluating state {idx+1}/{len(start_positions)}")
            estimated_value = self._sample(policy, state)
            value_map[state.y][state.x] = estimated_value
        return value_map

    def _sample(self, policy: EquiprobablePolicy, state: Position) -> float:
        total_reward = 0
        for i in range(self._nbr_samples):
            print(f"\tSample {i+1}/{self._nbr_samples}", end="\r") 
            episode_reward = 0
            self._env.set_state(state)
            current_state = state
            while not self._env.is_terminal_state(current_state):
                next_action = policy.pick_action(current_state)
                reward, current_state = self._env.do_action(next_action)
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / self._nbr_samples
