from Environment import Position, Action


class EquiprobablePolicy:

    def __init__(self):
        pass

    def pick_action(self, state: Position) -> Action:
        return Action.random()


class EpsilonGreedy:

    def __init__(self, epsilon: float = 0.05):
        self._epsilon = epsilon

    