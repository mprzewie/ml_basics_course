from typing import NamedTuple, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
from gym.envs.toy_text.discrete import DiscreteEnv
from matplotlib.animation import FuncAnimation

PL = 0
MNT = 1
WTCH = 2
MNST = 3

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [UP, RIGHT, DOWN, LEFT]

COLORS = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 1],
    [1, 0, 0]
]).astype(float)

DELTA = {
    UP: np.array([-1, 0]),
    RIGHT: np.array([0, 1]),
    DOWN: np.array([1, 0]),
    LEFT: np.array([0, -1]),
}


def map_from_csv(f_name: str) -> np.ndarray:
    return np.genfromtxt(f_name, delimiter=',').astype(int)


def setup_board(height: int, width: int) -> np.ndarray:
    board = np.zeros((height, width)).astype(int)
    board[0, :] = 1
    board[-1, :] = 1
    board[:, 0] = 1
    board[:, -1] = 1

    board[1, 4] = board[2, 2] = board[2, 3] = board[2, 4] = board[3, 3] = board[3, 4] = MNT
    # board[8,7] = board[7,7] = board[7,8] = MNT
    return board


class HyperParams(NamedTuple):
    exploration_rate: float = 0.0
    learning_rate: float = 0.0
    discount_rate: float = 0.0


class SARSA(NamedTuple):
    state_1: int
    action_1: int
    reward: float
    state_2: float
    action_2: float


class KaerMorhenv(DiscreteEnv):
    def __init__(
            self,
            board: np.ndarray = setup_board(10, 10),
            witcher_coords: np.ndarray = np.array([1, 1]),
            monster_coords: np.ndarray = np.array([8, 8])

    ):
        self.board: np.ndarray = board
        self.monster_coords = monster_coords
        self.witcher_hp = None

        n_states = np.prod(self.board.shape).sum()

        initial_state_distribution = np.zeros(n_states)
        initial_state_distribution[self.coords_to_state(witcher_coords)] = 1
        n_actions = len(ACTIONS)

        transition_probabilities = {}

        for state in range(n_states):
            y, x = self.state_to_coords(state)
            transition_probabilities[state] = {
                action: self._transition_probability_state_reward_done(np.array([y, x]), action)
                for action in ACTIONS

            }

        super().__init__(
            nS=n_states,
            nA=n_actions,
            P=transition_probabilities,
            isd=initial_state_distribution
        )
        self.reset()

    def reset(self):
        self.witcher_hp = 100

        return super(KaerMorhenv, self).reset()

    def step(self, a) -> Tuple[int, float, bool, Dict]:
        self.witcher_hp -= 1
        old_state = self.s
        old_done = self._done
        new_state, reward, done, info = super(KaerMorhenv, self).step(a)

        if old_done:
            self.s = new_state = old_state
        reward = self._reward if self._reward else reward
        done = self._done

        return new_state, reward, done, info

    def _render_state(self, ax):
        ax.imshow(COLORS[self._state_board])
        status = "Wiedźmin szuka potwora"
        if self.monster_dead:
            status = "Hurra! Wiedźmin wygrał!"
        elif self.witcher_dead:
            status = "To nie ma sensu. Idę pić."

        ax.set(xlabel=status)

        return ax

    def render(self, mode='human'):
        fig, ax = plt.subplots()
        return self._render_state(ax)

    def render_actions(self, actions: List[int], interval: int = 400):
        fig, ax = plt.subplots()
        self.reset()

        def update(i: int):
            if i > 0:
                self.step(actions[i - 1])
            self._render_state(ax)
            ax.set(title=f"Step {i}: coords={self.witcher_coords}")

        anim = FuncAnimation(fig, update, frames=range(len(actions) + 1), interval=interval)
        return anim

    @property
    def _state_board(self) -> np.ndarray:
        state_board = self.board.copy()
        state_board[tuple(self.monster_coords)] = MNST
        state_board[tuple(self.witcher_coords)] = WTCH
        return state_board

    def _transition_probability_state_reward_done(
            self, current_coords: np.ndarray, action: int
    ) -> List[Tuple[float, int, float, bool]]:
        delta = DELTA[action]
        new_coords = current_coords + delta
        c_y, c_x = current_coords
        n_y, n_x = new_coords
        probabilitity = 0.0 if self.board[c_y][c_x] == MNT or self.board[n_y][n_x] == MNT else 1.0
        if probabilitity > 0:
            new_state = self.coords_to_state(np.array([n_y, n_x]))
        else:
            new_state = self.coords_to_state(np.array([c_y, c_x]))

        done = new_state == self.coords_to_state(self.monster_coords)
        if done:
            reward = 100
        else:
            reward = -1
        return [(probabilitity, new_state, reward, done)]

    @property
    def witcher_coords(self):
        return self.state_to_coords(self.s)

    def coords_to_state(self, coords: np.ndarray) -> int:
        return np.ravel_multi_index(coords, self.board.shape).sum()

    def state_to_coords(self, state: int):
        return np.unravel_index(state, self.board.shape)

    @property
    def monster_dead(self) -> bool:
        return self.s == self.coords_to_state(self.monster_coords)

    @property
    def witcher_dead(self):
        return self.witcher_hp <= 0

    @property
    def _reward(self):
        if self.monster_dead:
            return 100
        if self.witcher_dead:
            return -100

    @property
    def _done(self):
        return self.monster_dead or self.witcher_dead
