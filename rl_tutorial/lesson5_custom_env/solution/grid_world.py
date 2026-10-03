"""Reference solution for Lesson 5."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rl_tutorial.lesson5_custom_env.grid_world import ARROWS, MOVES

__all__ = ["ARROWS", "MOVES", "GridWorldEnv", "potential_shaping"]


class GridWorldEnv(gym.Env):
    """A square grid. Start top-left, reach the goal bottom-right."""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        *,
        obstacles: tuple[tuple[int, int], ...] = (),
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if size < 2:
            raise ValueError(f"size must be at least 2, got {size}")

        self.size = int(size)
        self.obstacles = {tuple(o) for o in obstacles}
        self.render_mode = render_mode

        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("start and goal must not be obstacles")

        # Discrete observations so Lesson 2's tabular agent can index directly.
        self.observation_space = gym.spaces.Discrete(self.size * self.size)
        self.action_space = gym.spaces.Discrete(len(MOVES))

        self.agent = self.start

    def _get_obs(self) -> int:
        # A plain int: numpy integers fail `contains` in some versions.
        return int(self.agent[0] * self.size + self.agent[1])

    def _get_info(self) -> dict:
        return {
            "distance": self._manhattan(self.agent, self.goal),
            "agent": self.agent,
        }

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # FIRST: this seeds self.np_random.
        super().reset(seed=seed)

        if options and options.get("random_start"):
            # Draw from self.np_random, never the global np.random.
            while True:
                pos = (
                    int(self.np_random.integers(self.size)),
                    int(self.np_random.integers(self.size)),
                )
                if pos != self.goal and pos not in self.obstacles:
                    self.agent = pos
                    break
        else:
            self.agent = self.start

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")

        dr, dc = MOVES[action]
        candidate = (self.agent[0] + dr, self.agent[1] + dc)

        in_bounds = 0 <= candidate[0] < self.size and 0 <= candidate[1] < self.size
        if not in_bounds or candidate in self.obstacles:
            # Refuse the move: -1 for the step, -1 for hitting the wall.
            reward = -2.0
        else:
            self.agent = candidate
            reward = -1.0

        terminated = self.agent == self.goal
        if terminated:
            reward += 10.0

        # truncated stays False: the step limit belongs to the TimeLimit
        # wrapper that gym.make applies, not to the environment.
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ansi(self) -> str:
        rows = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.agent:
                    row += "A"
                elif (r, c) == self.goal:
                    row += "G"
                elif (r, c) in self.obstacles:
                    row += "#"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

    def _render_rgb(self) -> np.ndarray:
        cell = 32
        img = np.full((self.size * cell, self.size * cell, 3), 250, dtype=np.uint8)

        def fill(pos, color):
            r, c = pos
            img[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = color

        for obstacle in self.obstacles:
            fill(obstacle, (60, 60, 60))
        fill(self.goal, (27, 175, 122))
        fill(self.agent, (42, 120, 214))
        img[::cell, :, :] = 200
        img[:, ::cell, :] = 200
        return img


def potential_shaping(
    state: tuple[int, int],
    next_state: tuple[int, int],
    goal: tuple[int, int],
    gamma: float,
) -> float:
    """F(s, s') = gamma * Phi(s') - Phi(s), with Phi(s) = -distance(s, goal)."""

    def phi(pos: tuple[int, int]) -> float:
        return -float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    return gamma * phi(next_state) - phi(state)
