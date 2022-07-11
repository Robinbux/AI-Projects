import random
from dataclasses import dataclass
from enum import Enum

# Gridworld map
# 0 = free
# 1 = wall
# 2 = snakepit
# 3 = treasure
from typing import Tuple

map = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3]
]


@dataclass
class Position:
    x: int
    y: int


class Action(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

    @classmethod
    def random(cls):
        return random.choice(list(Action))


class Environment:

    def __init__(self) -> None:
        # Define map
        self._walls: list[Position] = []
        self._snakepit: Position
        self._treasure: Position
        self._read_map()

    def set_state(self, state: Position) -> None:
        self._agent_state = state

    def do_action(self, action: Action) -> Tuple[int, Position]:
        """ Do given action. Update the agent state based on it and return the reward
        and bool if the game is terminated"""
        new_state = self._agent_state
        match action:
            case Action.UP:
                new_state.x -= 1
            case Action.RIGHT:
                new_state.y += 1
            case Action.DOWN:
                new_state.x += 1
            case Action.LEFT:
                new_state.y -= 1
        if new_state.x in [-1, 9] or new_state.y in [-1, 9] or new_state in self._walls:
            # Ran against border or wall -> State doesn't change
            return -1, self._agent_state
        match new_state:
            case self._snakepit:
                return -50, self._agent_state
            case self._treasure:
                return 50, self._agent_state
            case _:
                self._agent_state = new_state
                return -1, self._agent_state

    def get_possible_start_states(self) -> list[Position]:
        states = []
        for i in range(len(map)):
            for j in range(len(map[i])):
                state = Position(i, j)
                if state not in self._walls and state != self._snakepit and state != self._treasure:
                    states.append(state)
        return states

    def is_terminal_state(self, state: Position) -> bool:
        return state == self._snakepit or state == self._treasure

    def _read_map(self) -> None:
        for i in range(len(map)):
            for j in range(len(map[i])):
                match map[i][j]:
                    case 1:
                        self._walls.append(Position(i, j))
                    case 2:
                        self._snakepit = Position(i, j)
                    case 3:
                        self._treasure = Position(i, j)
